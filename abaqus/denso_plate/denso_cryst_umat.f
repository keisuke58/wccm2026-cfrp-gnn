C =====================================================================
C  denso_cryst_umat.f  --  Abaqus/Standard UMAT for the Denso short-fibre
C  plate: orthotropic thermo-viscoelastic layer with NON-ISOTHERMAL
C  CRYSTALLIZATION (Nakamura law, Hoffman-Lauritzen K(T)).
C
C  Derived from ../cfrtp_cryst_umat_hl.f (run in Abaqus 2024, 2026-08-01).
C  ONE change: the crystallization shrinkage is ORTHOTROPIC,
C     dE*_ii = A_i dT + B_i dalpha   (i = 1, 2, 3),
C  because an orientation-averaged short-fibre layer shrinks differently
C  along x, y and z (the parent UMAT, written for a UD ply, applies a
C  single BETA to the two transverse directions only). B_i is the strain
C  per unit RELATIVE crystallinity alpha (= <beta>_ii * X_inf).
C  Everything else (HL kinetics, CHILE fraction, Prony/WLF/crystallinity
C  shift, tangent) is the parent code unchanged. With AGEL = 0, G0 = 1
C  the CHILE fraction is g = 1 (stiffness is not tied to crystallinity;
C  the glass transition enters through the WLF shift).
C
C  ***  NOT RUN IN ABAQUS YET.  Checked here only against a line-by-line ***
C  ***  Python port at a material point (gfortran driver), see           ***
C  ***  process_reference.py.                                            ***
C
C  PROPS (*USER MATERIAL, CONSTANTS=34) -- as the parent except 13, 33, 34:
C     1-3   E1,E2,E3        10-12 A1,A2,A3 (CTE, 1/K)
C     4-6   NU12,NU13,NU23  13    B1 (crystallization strain /dalpha, dir 1)
C     7-9   G12,G13,G23     14    NAVRAMI (n)
C                           15    K0 [1/s]   16 USTAR (U*/R) [K]
C                           17    KG [K^2]   18 TM0 [C]   19 TINF [C]
C                           20    ALPHAGEL 21 G0    22 GINF
C                           23-24 G1,TAU1 25-26 G2,TAU2  27-28 G3,TAU3
C                           29-31 WLF_C1, WLF_C2, WLF_TREF    32 BX
C                           33-34 B2, B3
C  State variables (DEPVAR = 23): as the parent.
C     STATEV(1)=alpha  (2)=g  (3)=max|s11|  (4)=a_T  (5)=a_X
C     STATEV(6..11)=q_1  (12..17)=q_2  (18..23)=q_3
C  PHYSICAL time [s]; temperatures in deg C (TABS added internally).
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
      PARAMETER (ZERO=0.D0, ONE=1.D0, HALF=0.5D0, TWO=2.D0, TEN=10.D0,
     1 TABS=273.15D0)
C
C ---- unpack properties -------------------------------------------------
      E1=PROPS(1); E2=PROPS(2); E3=PROPS(3)
      XNU12=PROPS(4); XNU13=PROPS(5); XNU23=PROPS(6)
      G12=PROPS(7); G13=PROPS(8); G23=PROPS(9)
      A1=PROPS(10); A2=PROPS(11); A3=PROPS(12)
      B1=PROPS(13); B2=PROPS(33); B3=PROPS(34)
      XN=PROPS(14)
      XK0=PROPS(15); USTAR=PROPS(16); XKG=PROPS(17)
      TM0=PROPS(18); TINF=PROPS(19)
      AGEL=PROPS(20); G0=PROPS(21); GINF=PROPS(22)
      GK(1)=PROPS(23); TAUK(1)=PROPS(24)
      GK(2)=PROPS(25); TAUK(2)=PROPS(26)
      GK(3)=PROPS(27); TAUK(3)=PROPS(28)
      WC1=PROPS(29); WC2=PROPS(30); WTREF=PROPS(31)
      BX=PROPS(32)
C
C ---- Hoffman-Lauritzen K(T) at the mid-increment temperature -----------
      TC=TEMP+HALF*DTEMP
      TKK=TC+TABS
      TM0K=TM0+TABS
      TINFK=TINF+TABS
      IF (TKK.GE.TM0K .OR. TKK.LE.TINFK) THEN
        XK=ZERO
      ELSE
        DTU=TM0K-TKK
        FF=TWO*TKK/(TM0K+TKK)
        AEXP=-USTAR/(TKK-TINFK)-XKG/(TKK*DTU*FF)
        IF (AEXP.LT.-700.D0) AEXP=-700.D0
        XK=XK0*EXP(AEXP)
      END IF
C
C ---- non-isothermal crystallization (Nakamura) over the increment ------
      ALPHA=STATEV(1)
      IF (ALPHA.LT.1.D-8) ALPHA=1.D-8
      IF (ALPHA.GT.ONE-1.D-10) ALPHA=ONE-1.D-10
      ARGL=-LOG(ONE-ALPHA)
      IF (ARGL.LT.1.D-12) ARGL=1.D-12
      POWL=(XN-ONE)/XN
      RATE=XN*XK*(ONE-ALPHA)*ARGL**POWL
      DALPHA=RATE*DTIME
      IF (DALPHA.LT.ZERO) DALPHA=ZERO
      ALPHA=ALPHA+DALPHA
      IF (ALPHA.GT.ONE) ALPHA=ONE
C
C ---- CHILE stiffness fraction g(alpha) ---------------------------------
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
      IF (DENOM.LT.ONE) DENOM=ONE
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
      DEIG(1)=A1*DTEMP+B1*DALPHA
      DEIG(2)=A2*DTEMP+B2*DALPHA
      IF (NTENS.GE.3) DEIG(3)=A3*DTEMP+B3*DALPHA
C
C ---- instantaneous (glassy) stress increment ---------------------------
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
