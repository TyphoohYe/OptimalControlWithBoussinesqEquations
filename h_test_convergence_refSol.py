
from Projection_Variation_con import Projection_Varion
from netgen.geom2d import SplineGeometry
import numpy as np
from netgen.occ import * 
from ngsolve import *

## load the ref sol

ref_opt_Y=np.load('np_optim_Y.npy')
ref_opt_Theta=np.load('np_optim_Theta.npy')
ref_opt_Mu=np.load('np_optim_Mu.npy')
ref_opt_Kappa=np.load('np_optim_Kappa.npy')
## set  step 

h=1/2**6
DeltaT=1/2**9

## set parameters
eta=1
alpha=0.1
u_a=-0.2
u_b=0.2

##
geo = SplineGeometry()

geo.AddRectangle((0, 0), (1, 1),bc='outer') 
mesh= Mesh(geo.GenerateMesh(maxh=h))

T_end=1
T_time=np.arange(0,T_end+DeltaT,DeltaT)
LenT=len(T_time)

Vel_Sp=VectorH1(mesh,order=1,dirichlet='outer')
Vel_Sp.SetOrder(TRIG,3)
Vel_Sp.Update()

PHeat_Sp=H1(mesh,order=1)

Sol_Y, Sol_Theta, Sol_Mu, Sol_Kappa=Projection_Varion(h,DeltaT)


mesh_ref = Mesh(geo.GenerateMesh(maxh=1/2**7))
Vel_SpRef=VectorH1(mesh_ref,order=1,dirichlet='outer')
Vel_SpRef.SetOrder(TRIG,3)
Vel_SpRef.Update()

PHeat_SpRef=H1(mesh_ref,order=1)

def Cal_errbound(cont1,cont2):
    PW_err=Integrate((cont1-cont2)**2, mesh_ref, definedon=mesh_ref.Boundaries('outer'), order=3)
    return PW_err 

Auxi_FunCoarVel=GridFunction(Vel_Sp)
Auxi_FunRefVel=GridFunction(Vel_SpRef)

Auxi_FunCoarHeat=GridFunction(PHeat_Sp)
Auxi_FunRefHeat=GridFunction(PHeat_SpRef)

np_Yerr_inf=np.zeros(LenT)
np_Thetaerr_inf=np.zeros(LenT)

np_Yerr_H1=np.zeros(LenT)
np_Thetaerr_H1=np.zeros(LenT)

for i in range(LenT):
   Auxi_FunCoarVel.vec.data=Sol_Y.vecs[i]
   Auxi_FunCoarHeat.vec.data=Sol_Theta.vecs[i]

   Auxi_FunRefVel.vec.data=np.array(ref_opt_Y[:,i])
   Auxi_FunRefHeat.vec.data=np.array(ref_opt_Theta[:,i])
   
   np_Yerr_inf[i]=Integrate((Auxi_FunCoarVel-Auxi_FunRefVel)**2, mesh_ref, order=3)
   np_Thetaerr_inf[i]=Integrate((Auxi_FunCoarHeat-Auxi_FunRefHeat)**2, mesh_ref, order=3)

   np_Yerr_H1[i]=Integrate(InnerProduct(Grad(Auxi_FunCoarVel)-Grad(Auxi_FunRefVel), Grad(Auxi_FunCoarVel)-Grad(Auxi_FunRefVel)), mesh_ref, order=3)
   np_Thetaerr_H1[i]=np_Thetaerr_inf[i]+Integrate((grad(Auxi_FunCoarHeat)-grad(Auxi_FunRefHeat))**2, mesh_ref, order=3)
   

   
err_Y_inf=sqrt(np_Yerr_inf.max())
err_Theta_inf=sqrt(np_Thetaerr_inf.max())

err_Y_H1=sqrt(DeltaT*np_Yerr_H1.sum())
err_Theta_H1=sqrt(DeltaT*np_Thetaerr_H1.sum())


print('y-infy, theta-infty, y-H1, theta-H1', [err_Y_inf,err_Theta_inf, err_Y_H1, err_Theta_H1] )


np_Muerr_inf=np.zeros(LenT-1)
np_Kappaerr_inf=np.zeros(LenT-1)

np_Muerr_H1=np.zeros(LenT-1)
np_Kappaerr_H1=np.zeros(LenT-1)

for i in range(LenT-1):
   Auxi_FunCoarVel.vec.data=Sol_Mu.vecs[i]
   Auxi_FunCoarHeat.vec.data=Sol_Kappa.vecs[i]

   Auxi_FunRefVel.vec.data=np.array(ref_opt_Mu[:,i])
   Auxi_FunRefHeat.vec.data=np.array(ref_opt_Kappa[:,i])
   
   np_Muerr_inf[i]=Integrate((Auxi_FunCoarVel-Auxi_FunRefVel)**2, mesh_ref, order=3)
   np_Kappaerr_inf[i]=Integrate((Auxi_FunCoarHeat-Auxi_FunRefHeat)**2, mesh_ref, order=3)

   np_Muerr_H1[i]=Integrate(InnerProduct(Grad(Auxi_FunCoarVel)-Grad(Auxi_FunRefVel), Grad(Auxi_FunCoarVel)-Grad(Auxi_FunRefVel)), mesh_ref, order=3)
   np_Kappaerr_H1[i]=np_Kappaerr_inf[i]+Integrate((grad(Auxi_FunCoarHeat)-grad(Auxi_FunRefHeat))**2, mesh_ref, order=3)
   

   
err_Mu_inf=sqrt(np_Muerr_inf.max())
err_Kappa_inf=sqrt(np_Kappaerr_inf.max())

err_Mu_H1=sqrt(DeltaT*np_Muerr_H1.sum())
err_Kappa_H1=sqrt(DeltaT*np_Kappaerr_H1.sum())


print('mu-infy, kappa-infty, mu-H1, kappa-H1', [err_Mu_inf,err_Kappa_inf, err_Mu_H1, err_Kappa_H1] )

Auxi_FunCoarse=GridFunction(PHeat_Sp)
Auxi_FunRefmesh=GridFunction(PHeat_SpRef)
Sum_Q=np.zeros(LenT-1)

for  k in range(LenT-1):
    Auxi_FunCoarse.vec.data=-eta/alpha*Sol_Kappa.vecs[LenT-k-2]
    Cont_n=IfPos(Auxi_FunCoarse-u_b,u_b,0)+IfPos(Auxi_FunCoarse-u_a,0,u_a)+IfPos((Auxi_FunCoarse-u_b)*(Auxi_FunCoarse-u_a),0,Auxi_FunCoarse)
               
    Auxi_FunRefmesh.vec.data=np.array(-eta/alpha*np.array(ref_opt_Kappa[:,LenT-k-2]))
    Cont_Nminus1=IfPos(Auxi_FunRefmesh-u_b,u_b,0)+IfPos(Auxi_FunRefmesh-u_a,0,u_a)+IfPos((Auxi_FunRefmesh-u_b)*(Auxi_FunRefmesh-u_a),0,Auxi_FunRefmesh)
    Sum_Q[k]=Cal_errbound(Cont_Nminus1,Cont_n)


Err_Q=sqrt(DeltaT*Sum_Q.sum())

print("Control err: ",Err_Q)
