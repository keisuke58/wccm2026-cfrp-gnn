C =====================================================================
C  cfrtp_cryst_umat_ve.f  --  Abaqus/Standard UMAT SKELETON
C  Semi-crystalline thermoplastic CFRTP (Daikin fluoropolymer matrix):
C  orthotropic thermo-VISCOELASTIC ply whose stiffness, shrinkage and
C  relaxation are driven by NON-ISOTHERMAL CRYSTALLIZATION KINETICS
C  (Nakamura model).  This is the "semi-crystalline" upgrade of
C  cfrtp_cure_umat_ve.f: instead of Arrhenius cure, the state alpha is the
C  RELATIVE CRYSTALLINITY, developed on COOLING through the crystallization
C  window, and it controls
C     - stiffness development  g(alpha)          (CHILE-style)
C     - crystallization shrinkage eigenstrain    (BETA*dalpha)
C     - relaxation-time crystallinity shift a_X  (crystals pin relaxation)
C  Reproduces the cooling-rate -> crystallinity -> residual-stress pathway
C  of the Python demo cfrtp_residual_stress_fe.py, with viscoelastic
C  relaxation (cf. cfrtp_viscoelastic_residual_stress.py) on top.
C
C  ***  SKELETON -- NOT VERIFIED IN ABAQUS (no license in this env).      ***
C  ***  The scheme is numerically checked in Python; still run the        ***
C  ***  1-element sanity test and co-calibrate Nakamura K(T), Prony tau_k  ***
C  ***  and the WLF/crystallinity shifts against DMA + residual-stress     ***
C  ***  measurements before trusting magnitudes.                          ***
C
C  Crystallization (Nakamura non-isothermal Avrami rate form):
C     dalpha/dt = n K(T) (1-alpha) [-ln(1-alpha)]^((n-1)/n)
C     K(T) = KMAX exp[ -((T-TCRYST)/WCRYST)^2 ]   (bell-shaped window;
C            swap for Hoffman-Lauritzen / true Nakamura K(T) if desired)
C  Seed alpha0 ~ 1e-3 via *INITIAL CONDITIONS, TYPE=SOLUTION (nucleation).
C
C  Viscoelasticity (generalized Maxwell / Prony, incremental partial
C  stresses q_k, same as cfrtp_cure_umat_ve.f) with shifted time
C     tau_k*(T,alpha) = a_T(T) * a_X(alpha) * tau_k,
C     log10 a_T = -C1 (T-Tref)/(C2 + (T-Tref))      (WLF, temperature)
C     a_X       = 10^( BX * alpha )                 (crystallinity: freezes
C                                                    relaxation as alpha->1)
C
C  State variables (DEPVAR = 23):
C     STATEV(1)      = relative crystallinity alpha  [0,1]
C     STATEV(2)      = stiffness fraction g(alpha)    (output)
C     STATEV(3)      = max |sigma_11| history         (output)
C     STATEV(4)      = a_T (WLF shift)                (output)
C     STATEV(5)      = a_X (crystallinity shift)      (output)
C     STATEV(6..11)  = q_1 (6 comp)   Maxwell branch partial stresses
C     STATEV(12..17) = q_2 (6 comp)
C     STATEV(18..23) = q_3 (6 comp)
C
C  PROPS (*USER MATERIAL, CONSTANTS=30):
C     1-3   E1,E2,E3        10-12 A1,A2,A3 (CTE, 1/K)
C     4-6   NU12,NU13,NU23  13    BETA (crystallization shrinkage /dalpha)
C     7-9   G12,G13,G23     14    NAVRAMI (Avrami/Nakamura exponent n)
C                           15    KMAX     16 TCRYST  17 WCRYST  (K(T) window)
C                           18    ALPHAGEL 19 G0      (CHILE g(alpha))
C                           20    GINF
C                           21-22 G1,TAU1  23-24 G2,TAU2  25-26 G3,TAU3
C                           27-29 WLF_C1, WLF_C2, WLF_TREF
C                           30    BX (crystallinity relaxation-shift exponent)
C  Drive with a MELT->COOL temperature cycle via *TEMPERATURE (TEMP/DTEMP);
C  TCRYST/TREF are in the same temperature unit as TEMP (e.g. deg C).
C =====================================================================
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     2 PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,
     3 DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)
C
      DIMENSION SC(6,6),CC(6,6),DEIG(6),DSINST(6)
      DIMENSION Q(6,3),SUMQ(6),GK(3),TAUK(3),BETAK(3),EXPK(3)
      PARAMETER (ZERO=0.D0, ONE=1.D0, HALF=0.5D0, TEN=10.D0)
C
C ---- unpack properties -------------------------------------------------
      E1=PROPS(1); E2=PROPS(2); E3=PROPS(3)
      XNU12=PROPS(4); XNU13=PROPS(5); XNU23=PROPS(6)
      G12=PROPS(7); G13=PROPS(8); G23=PROPS(9)
      A1=PROPS(10); A2=PROPS(11); A3=PROPS(12)
      BETA=PROPS(13)
      XN=PROPS(14); XKMAX=PROPS(15); TCRYST=PROPS(16); WCRYST=PROPS(17)
      AGEL=PROPS(18); G0=PROPS(19)
      GINF=PROPS(20)
      GK(1)=PROPS(21); TAUK(1)=PROPS(22)
      GK(2)=PROPS(23); TAUK(2)=PROPS(24)
      GK(3)=PROPS(25); TAUK(3)=PROPS(26)
      WC1=PROPS(27); WC2=PROPS(28); WTREF=PROPS(29)
      BX=PROPS(30)
C
C ---- non-isothermal crystallization (Nakamura) over the increment ------
      ALPHA=STATEV(1)
      IF (ALPHA.LT.1.D-8) ALPHA=1.D-8
      IF (ALPHA.GT.ONE-1.D-10) ALPHA=ONE-1.D-10
      TC=TEMP+HALF*DTEMP
      ARGK=-((TC-TCRYST)/WCRYST)**2
      IF (ARGK.LT.-60.D0) ARGK=-60.D0
      XK=XKMAX*EXP(ARGK)
      ARGL=-LOG(ONE-ALPHA)
      IF (ARGL.LT.1.D-12) ARGL=1.D-12
      POWL=(XN-ONE)/XN
      RATE=XN*XK*(ONE-ALPHA)*ARGL**POWL
      DALPHA=RATE*DTIME
      IF (DALPHA.LT.ZERO) DALPHA=ZERO
      ALPHA=ALPHA+DALPHA
      IF (ALPHA.GT.ONE) ALPHA=ONE
C
C ---- CHILE stiffness fraction g(alpha): 0 below percolation -> 1 solid --
      X=(ALPHA-AGEL)/(ONE-AGEL)
      IF (X.LT.ZERO) X=ZERO
      IF (X.GT.ONE)  X=ONE
      G=G0+(ONE-G0)*X
C
C ---- orthotropic compliance (order 11,22,33,12,13,23) -> stiffness ------
      DO I=1,6
        DO J=1,6
          SC(I,J)=ZERO
        END DO
      END DO
      SC(1,1)=ONE/E1;  SC(2,2)=ONE/E2;  SC(3,3)=ONE/E3
      SC(1,2)=-XNU12/E1; SC(2,1)=SC(1,2)
      SC(1,3)=-XNU13/E1; SC(3,1)=SC(1,3)
      SC(2,3)=-XNU23/E2; SC(3,2)=SC(2,3)
      SC(4,4)=ONE/G12; SC(5,5)=ONE/G13; SC(6,6)=ONE/G23
      CALL KINV6(SC,CC)
C
C ---- relaxation shifts: WLF (temperature) x crystallinity --------------
      TMID=TEMP+HALF*DTEMP
      DTT=TMID-WTREF
      DENOM=WC2+DTT
      IF (ABS(DENOM).LT.ONE) DENOM=SIGN(ONE,DENOM)
      POW=-WC1*DTT/DENOM
      IF (POW.GT. 30.D0) POW= 30.D0
      IF (POW.LT.-30.D0) POW=-30.D0
      AT=TEN**POW
      POWX=BX*ALPHA
      IF (POWX.GT.30.D0) POWX=30.D0
      AX=TEN**POWX
      ASHIFT=AT*AX
C
C ---- per-branch relaxation factors (shifted time tau_k* = ASHIFT*tau_k)--
      DO K=1,3
        TAUS=ASHIFT*TAUK(K)
        IF (TAUS.LT.1.D-20) TAUS=1.D-20
        XR=DTIME/TAUS
        EXPK(K)=EXP(-XR)
        IF (XR.GT.1.D-6) THEN
          BETAK(K)=(ONE-EXPK(K))/XR
        ELSE
          BETAK(K)=ONE-HALF*XR
        END IF
      END DO
C
C ---- incremental eigenstrain: thermal + crystallization shrinkage ------
      DO I=1,NTENS
        DEIG(I)=ZERO
      END DO
      DEIG(1)=A1*DTEMP
      DEIG(2)=A2*DTEMP+BETA*DALPHA
      IF (NTENS.GE.3) DEIG(3)=A3*DTEMP+BETA*DALPHA
C
C ---- instantaneous (glassy) stress increment dsig_inst = C_inst:(de-deig)
      DO I=1,NTENS
        DSINST(I)=ZERO
        DO J=1,NTENS
          DSINST(I)=DSINST(I)+G*CC(I,J)*(DSTRAN(J)-DEIG(J))
        END DO
      END DO
C
C ---- load branch partial stresses q_k^n and their sum ------------------
      DO I=1,NTENS
        SUMQ(I)=ZERO
      END DO
      DO K=1,3
        DO I=1,NTENS
          Q(I,K)=STATEV(5+(K-1)*6+I)
          SUMQ(I)=SUMQ(I)+Q(I,K)
        END DO
      END DO
C
C ---- generalized-Maxwell update: sigma = q_inf + sum_k q_k --------------
      DO I=1,NTENS
        QINF=STRESS(I)-SUMQ(I)
        QINF=QINF+GINF*DSINST(I)
        SNEW=QINF
        DO K=1,3
          Q(I,K)=EXPK(K)*Q(I,K)+GK(K)*BETAK(K)*DSINST(I)
          SNEW=SNEW+Q(I,K)
        END DO
        STRESS(I)=SNEW
      END DO
C
C ---- consistent tangent: DDSDDE = (g_inf + sum g_k beta_k) * C_inst ----
      GTAN=GINF
      DO K=1,3
        GTAN=GTAN+GK(K)*BETAK(K)
      END DO
      DO I=1,NTENS
        DO J=1,NTENS
          DDSDDE(I,J)=GTAN*G*CC(I,J)
        END DO
      END DO
C
C ---- store state -------------------------------------------------------
      STATEV(1)=ALPHA
      STATEV(2)=G
      STATEV(4)=AT
      STATEV(5)=AX
      DO K=1,3
        DO I=1,NTENS
          STATEV(5+(K-1)*6+I)=Q(I,K)
        END DO
      END DO
      IF (NSTATV.GE.3) THEN
        IF (ABS(STRESS(1)).GT.STATEV(3)) STATEV(3)=ABS(STRESS(1))
      END IF
C
      RETURN
      END
C =====================================================================
C  KINV6 : inverse of a 6x6 matrix by Gauss-Jordan (skeleton helper)
C =====================================================================
      SUBROUTINE KINV6(A,AINV)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION A(6,6),AINV(6,6),W(6,12)
      PARAMETER (ZERO=0.D0, ONE=1.D0)
      DO I=1,6
        DO J=1,6
          W(I,J)=A(I,J)
          W(I,J+6)=ZERO
        END DO
        W(I,I+6)=ONE
      END DO
      DO I=1,6
        P=W(I,I)
        DO J=1,12
          W(I,J)=W(I,J)/P
        END DO
        DO K=1,6
          IF (K.NE.I) THEN
            F=W(K,I)
            DO J=1,12
              W(K,J)=W(K,J)-F*W(I,J)
            END DO
          END IF
        END DO
      END DO
      DO I=1,6
        DO J=1,6
          AINV(I,J)=W(I,J+6)
        END DO
      END DO
      RETURN
      END
