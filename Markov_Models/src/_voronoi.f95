!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! VORONOI INITIALIZATION
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Generates a voronoi space from given centroids, assigned previously or
! obtained from kmeans/PCCA+ methods.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SUBROUTINE INITIALIZE(States,data,centroids,clusters,PBC,ncentroids,nfeatures,nsamples)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  VARIABLES
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

IMPLICIT NONE
INTEGER,INTENT(in) :: nsamples,ncentroids,nfeatures
INTEGER,INTENT(in) :: clusters(ncentroids), PBC(nfeatures)
DOUBLE PRECISION, INTENT(in) :: centroids(ncentroids,nfeatures),data(nsamples,nfeatures)
INTEGER(kind=8),INTENT(inout) :: States(nsamples)

INTEGER :: i,j,k,n_min(ncentroids)
DOUBLE PRECISION :: cost(ncentroids),D(ncentroids,nfeatures)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  CLUSTERING BY MINIMIZED COST
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! The cost is calculated as the euclidean distance between the randomized
! centroids and each point in the index-space, States. States maps one-to-one to the
! coordinate space, Q.

D = 0.d0
DO i=1,nsamples
    DO j=1,ncentroids
        DO k=1,nfeatures
            IF(PBC(k)==1)THEN
                IF(ABS(data(i,k)-centroids(j,k))>0.5)THEN
                    IF(data(i,k)>centroids(j,k))THEN
                        D(j,k) = (ABS(1-data(i,k))+centroids(j,k))**2
                    ELSE IF(centroids(j,k)>=data(i,k))THEN
                        D(j,k) = (ABS(1-centroids(j,k))+data(i,k))**2
                    END IF
                ELSE
                    D(j,k) = (data(i,k)-centroids(j,k))**2
                END IF
            ELSE IF(PBC(k)==0)THEN
                D(j,k) = (data(i,k)-centroids(j,k))**2
            END IF
        END DO
        cost(j) = DSQRT(SUM(D(j,:)))
    END DO
    n_min = MINLOC(cost,DIM=1)
    States(i) = clusters(n_min(1))
END DO
END
