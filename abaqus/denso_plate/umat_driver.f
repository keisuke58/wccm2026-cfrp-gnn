C =====================================================================
C  umat_driver.f -- drive denso_cryst_umat.f at ONE material point with a
C  prescribed history (time increment, temperature, temperature increment,
C  6 strain increments per line), so that the Fortran UMAT can be compared
C  with its Python port (process_reference.py --check, P1) without Abaqus.
C  Build (process_reference.py does this):
C    gfortran -I<dir with ABA_PARAM.INC> -o umat_driver umat_driver.f \
C             denso_cryst_umat.f
C  Output per increment: STRESS(1..6), STATEV(1), STATEV(6..23).
C =====================================================================
      PROGRAM DRIVER
      INCLUDE 'ABA_PARAM.INC'
      PARAMETER (NTENS=6, NSTATV=23, NPROPS=34)
      CHARACTER*80 CMNAME
      CHARACTER*256 FNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)
      CALL GET_COMMAND_ARGUMENT(1, FNAME)
      OPEN(10, FILE=FNAME, STATUS='OLD')
      READ(10,*) NSTEP
      READ(10,*) (PROPS(I), I=1,NPROPS)
      DO I=1,NTENS
        STRESS(I)=0.D0
        STRAN(I)=0.D0
      END DO
      DO I=1,NSTATV
        STATEV(I)=0.D0
      END DO
      STATEV(1)=1.D-3
      TIME(1)=0.D0
      TIME(2)=0.D0
      CMNAME='DENSO'
      DO N=1,NSTEP
        READ(10,*) DTIME, TEMP, DTEMP, (DSTRAN(I), I=1,NTENS)
        CALL UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1   RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     2   PREDEF,DPRED,CMNAME,3,3,NTENS,NSTATV,PROPS,NPROPS,COORDS,
     3   DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,1,1,0,0,1,N)
        DO I=1,NTENS
          STRAN(I)=STRAN(I)+DSTRAN(I)
        END DO
        TIME(1)=TIME(1)+DTIME
        TIME(2)=TIME(2)+DTIME
        WRITE(*,'(25ES25.16E3)') (STRESS(I),I=1,NTENS), STATEV(1),
     1   (STATEV(I),I=6,23)
      END DO
      CLOSE(10)
      END
