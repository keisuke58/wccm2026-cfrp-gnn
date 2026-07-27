C =====================================================================
C  cfrtp_cure_umat.f  --  Abaqus/Standard UMAT SKELETON
C  Cure-hardening (CHILE) orthotropic ply with cure-kinetics, cure
C  (crystallization) shrinkage and thermal eigenstrain -- the Abaqus
C  counterpart of the Python demos cfrp_cure_residual_stress_fe.py (⑳)
C  and cfrtp_residual_stress_fe.py.  Builds residual stress incrementally
C  as the resin cures/solidifies and cools.
C
C  ***  SKELETON -- NOT VERIFIED IN ABAQUS (no license in this env).  ***
C  ***  Adapt PROPS/ordering, check sign conventions, run a 1-element  ***
C  ***  free-contraction test (should give ~0 residual stress) first.  ***
C
C  State variables (DEPVAR = 3 recommended):
C     STATEV(1) = degree of cure / solidification  alpha  [0,1]
C     STATEV(2) = CHILE stiffness fraction g(alpha)      (output)
C     STATEV(3) = max |sigma_11| history                (optional output)
C
C  PROPS (define with *USER MATERIAL, CONSTANTS=21):
C     1-3   E1,E2,E3        9-11  A1,A2,A3 (CTE, 1/K)
C     4-6   NU12,NU13,NU23  12    BETA  (transverse cure/cryst. shrink /dalpha)
C     7-9   G12,G13,G23     13-15 AK,EAK,NEXP (Arrhenius nth-order kinetics)
C                           16    ALPHAGEL (gel: stress builds above this)
C                           17    G0        (ungelled stiffness fraction)
C                           18    RGAS      (8.314)  19  TABS (273.15 if TEMP in C)
C  Thermal/cure driving: supply the cure-cycle temperature via *TEMPERATURE
C  (predefined field, TEMP/DTEMP) so the kinetics integrate along the cycle.
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
      DIMENSION SC(6,6),CC(6,6),DEIG(6)
      PARAMETER (ZERO=0.D0, ONE=1.D0)
C
C ---- unpack properties -------------------------------------------------
      E1=PROPS(1); E2=PROPS(2); E3=PROPS(3)
      XNU12=PROPS(4); XNU13=PROPS(5); XNU23=PROPS(6)
      G12=PROPS(7); G13=PROPS(8); G23=PROPS(9)
      A1=PROPS(10); A2=PROPS(11); A3=PROPS(12)
      BETA=PROPS(13)
      AK=PROPS(14); EAK=PROPS(15); XN=PROPS(16)
      AGEL=PROPS(17); G0=PROPS(18); RG=PROPS(19); TABS=PROPS(20)
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
C ---- tangent DDSDDE = g * C (scale to current cure state) --------------
      DO I=1,NTENS
        DO J=1,NTENS
          DDSDDE(I,J)=G*CC(I,J)
        END DO
      END DO
C
C ---- incremental eigenstrain: thermal + cure/cryst. shrinkage ----------
C      (fibre direction ~ no shrink; transverse carries BETA*dalpha)
      DO I=1,NTENS
        DEIG(I)=ZERO
      END DO
      DEIG(1)=A1*DTEMP
      DEIG(2)=A2*DTEMP+BETA*DALPHA
      IF (NTENS.GE.3) DEIG(3)=A3*DTEMP+BETA*DALPHA
C
C ---- incremental stress update: dsig = C:(deps - deig) -----------------
      DO I=1,NTENS
        DSIG=ZERO
        DO J=1,NTENS
          DSIG=DSIG+DDSDDE(I,J)*(DSTRAN(J)-DEIG(J))
        END DO
        STRESS(I)=STRESS(I)+DSIG
      END DO
C
C ---- store state -------------------------------------------------------
      STATEV(1)=ALPHA
      STATEV(2)=G
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
