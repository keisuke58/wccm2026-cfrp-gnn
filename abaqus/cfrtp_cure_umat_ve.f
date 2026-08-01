C =====================================================================
C  cfrtp_cure_umat_ve.f  --  Abaqus/Standard UMAT SKELETON
C  Orthotropic CHILE (cure/solidification hardening) ply WITH thermo-
C  viscoelastic stress relaxation (generalized Maxwell / Prony series
C  + WLF temperature shift).  This is the "measurement-grade" extension
C  of cfrtp_cure_umat.f: hot -> fast relaxation (built-up stress bleeds
C  off), cold -> frozen (residual stress locks in).  Abaqus counterpart
C  of the Python demo cfrtp_viscoelastic_residual_stress.py
C  (elastic 500 -> viscoelastic 131 MPa, ~74% relaxation).
C
C  ***  SKELETON -- NOT VERIFIED IN ABAQUS (no license in this env).  ***
C  ***  Check ordering / sign conventions and co-calibrate the Prony   ***
C  ***  tau_k with the cure-cycle DURATION (they are in the same time   ***
C  ***  unit as the step) before trusting magnitudes.                   ***
C
C  Model (reduced relaxation applied to the instantaneous CHILE stiffness
C  C_inst = g(alpha)*C0, generalized-Maxwell incremental form):
C     sigma^{n+1} = q_inf^{n+1} + sum_k q_k^{n+1}
C     dsig_inst   = C_inst : (deps - deps_eig)
C     q_inf^{n+1} = (sigma^n - sum_k q_k^n) + g_inf * dsig_inst
C     q_k^{n+1}   = exp(-dt/tau_k*) q_k^n + g_k*beta_k*dsig_inst
C     beta_k      = (1 - exp(-dt/tau_k*)) / (dt/tau_k*)     (exact for
C                   strain linear-in-time over the increment)
C     tau_k*      = a_T * tau_k         (WLF shift; hot -> small -> relax)
C  Requires g_inf + sum_k g_k = 1 (normalized relaxation function).
C
C  State variables (DEPVAR = 22):
C     STATEV(1)      = degree of cure / solidification alpha  [0,1]
C     STATEV(2)      = CHILE stiffness fraction g(alpha)      (output)
C     STATEV(3)      = max |sigma_11| history                 (output)
C     STATEV(4)      = a_T shift factor at this point         (output)
C     STATEV(5..10)  = q_1 (6 comp)   Maxwell branch partial stresses
C     STATEV(11..16) = q_2 (6 comp)
C     STATEV(17..22) = q_3 (6 comp)
C
C  PROPS (*USER MATERIAL, CONSTANTS=30):
C     1-3   E1,E2,E3        10-12 A1,A2,A3 (CTE, 1/K)
C     4-6   NU12,NU13,NU23  13    BETA (transverse cure/cryst. shrink /dalpha)
C     7-9   G12,G13,G23     14-16 AK,EAK,NEXP (Arrhenius nth-order kinetics)
C                           17    ALPHAGEL   18 G0  19 RGAS  20 TABS
C                           21    GINF (relaxed fraction g_inf)
C                           22-23 G1,TAU1     24-25 G2,TAU2   26-27 G3,TAU3
C                           28-30 WLF_C1, WLF_C2, WLF_TREF  (Tref in same
C                                 temperature unit as TEMP, e.g. deg C)
C  Drive with the cure-cycle temperature via *TEMPERATURE (TEMP/DTEMP).
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
      AK=PROPS(14); EAK=PROPS(15); XN=PROPS(16)
      AGEL=PROPS(17); G0=PROPS(18); RG=PROPS(19); TABS=PROPS(20)
      GINF=PROPS(21)
      GK(1)=PROPS(22); TAUK(1)=PROPS(23)
      GK(2)=PROPS(24); TAUK(2)=PROPS(25)
      GK(3)=PROPS(26); TAUK(3)=PROPS(27)
      WC1=PROPS(28); WC2=PROPS(29); WTREF=PROPS(30)
C
C ---- cure kinetics: integrate alpha over the increment ------------------
      ALPHA=STATEV(1)
      TK=TEMP+DTEMP+TABS
      IF (TK.LT.ONE) TK=ONE
      RATE=AK*EXP(-EAK/(RG*TK))*(ONE-ALPHA)**XN
      DALPHA=RATE*DTIME
      IF (DALPHA.LT.ZERO) DALPHA=ZERO
      ALPHA=ALPHA+DALPHA
      IF (ALPHA.GT.ONE) ALPHA=ONE
C
C ---- CHILE stiffness fraction g(alpha): 0 below gel -> 1 solid ----------
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
C ---- WLF temperature shift a_T at the mid-increment temperature --------
C      log10(a_T) = -C1 (T-Tref) / (C2 + (T-Tref)) ; hot -> a_T<1 (fast),
C      cold -> a_T>>1 (frozen).  Clamp the exponent to avoid overflow.
      TMID=TEMP+HALF*DTEMP
      DTT=TMID-WTREF
      DENOM=WC2+DTT
      IF (ABS(DENOM).LT.ONE) DENOM=SIGN(ONE,DENOM)
      POW=-WC1*DTT/DENOM
      IF (POW.GT. 30.D0) POW= 30.D0
      IF (POW.LT.-30.D0) POW=-30.D0
      AT=TEN**POW
C
C ---- per-branch relaxation factors (shifted time tau_k* = a_T*tau_k) ----
      DO K=1,3
        TAUS=AT*TAUK(K)
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
C ---- incremental eigenstrain: thermal + cure/cryst. shrinkage ----------
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
          Q(I,K)=STATEV(4+(K-1)*6+I)
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
      DO K=1,3
        DO I=1,NTENS
          STATEV(4+(K-1)*6+I)=Q(I,K)
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
