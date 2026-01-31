from ngsolve import *
from netgen.geom2d import SplineGeometry
from State_NsHeat import state_NSHEAT
from Adjoint_NsHeat import adjoint_NSHEAT
from ngsolve.webgui import  Draw
import numpy as np

def Projection_Varion(h,DeltaT):
     geo = SplineGeometry()

     geo.AddRectangle((0, 0), (1, 1),bc='outer') 

     mesh = Mesh(geo.GenerateMesh(maxh=h))


     T_end=1
     # time interval
     T_time=np.arange(0,T_end+DeltaT,DeltaT)

     LenT=len(T_time)
     # define the time variable
     t=Parameter(0)
     eta=1
     alpha=0.1
     u_a=-0.2
     u_b=0.2
     
     #ex_Yini=CF((0,0))
     ex_Theta0=CF(0)


     #ex_Yd=CF((cos(pi*t)*cos(4*pi*x)*sin(4*pi*y), cos(pi*t)*sin(4*pi*x)*cos(4*pi*y)))
     ex_Yd=CF((50*x**2*(x-1)**2*(2*y*(y-1)**2+2*y**2*(y-1)), -50*y**2*(y-1)**2*(2*x*(x-1)**2+2*x**2*(x-1))))
     #ex_Yd=CF((100*x**2*(x-1)**2*(2*y*(y-1/3)**2+2*y**2*(y-1/3)), -100*y**2*(y-1/3)**2*(2*x*(x-1)**2+2*x**2*(x-1))))
     ex_Yini=CF((0,0))

     #ex_Yini=ex_Yd

     ex_ThetaD=CF(0)

     R_ns=CF((0,0))

     R_heat=CF(0)


     Vel_Sp=VectorH1(mesh,order=1,dirichlet='outer')
     Vel_Sp.SetOrder(TRIG,3)
     Vel_Sp.Update()

     PHeat_Sp=H1(mesh,order=1)

     Bound_Sp=Compress(H1(mesh,order=1, definedon=mesh.Boundaries('outer')))
     bound_Fun=GridFunction(Bound_Sp)
     Lenh_bound=len(bound_Fun.vec.data)

     np_XY=np.zeros((Lenh_bound,2))
     for i in range(Lenh_bound):
        meshv = mesh[NodeId(VERTEX,i)]
        np_XY[i,0]=meshv.point[0]
        np_XY[i,1]=meshv.point[1]
     

     ## 
     GridYd=GridFunction(Vel_Sp)
     GridYd.Interpolate(ex_Yd)

     vtk_Y= VTKOutput(mesh,coefs=[GridYd],names=["Yd"],filename="Yd",subdivision=2)
     vtk_Y.Do()
     ## projection the initial Y0
     fes_VelP=Vel_Sp*PHeat_Sp

     Auxi_Sol=GridFunction(fes_VelP)

     (u,p), (v, q)=fes_VelP.TnT()

     a_Stoke=BilinearForm(fes_VelP)
     a_Stoke+=u*v*dx-p*div(v)*dx-div(u)*q*dx

     b_Stoke=LinearForm(fes_VelP)
     b_Stoke+=ex_Yini*v*dx

     a_Stoke.Assemble()

     b_Stoke.Assemble()


     Auxi_Sol.vec.data=a_Stoke.mat.Inverse(fes_VelP.FreeDofs())*b_Stoke.vec

     Init_Y=GridFunction(Vel_Sp)
     Init_Y.vec.data=Auxi_Sol.components[0].vec.data

     ## projection the initial theta0
     mu, psi=PHeat_Sp.TnT()
     a_Mass=BilinearForm(PHeat_Sp)
     a_Mass+=mu*psi*dx

     b_Right=LinearForm(PHeat_Sp)
     b_Right+=ex_Theta0*psi*dx
     a_Mass.Assemble()
     b_Right.Assemble()

     Init_Theta=GridFunction(PHeat_Sp)
     Init_Theta.vec.data=a_Mass.mat.Inverse()*b_Right.vec
     ##

     Sol_Y=GridFunction(Vel_Sp, multidim=LenT)
     Sol_Theta=GridFunction(PHeat_Sp, multidim=LenT)


     Sol_Kappa=GridFunction(PHeat_Sp,multidim=LenT-1)
     Auxi_Sol_Kappa=GridFunction(PHeat_Sp,multidim=LenT-1)

     MaxIter=200

     mesh_Auxi = Mesh(geo.GenerateMesh(maxh=h))

     def Cal_err(cont1,cont2):
          PW_err=Integrate((cont1-cont2)**2, mesh_Auxi, definedon=mesh_Auxi.Boundaries('outer'), order=3)
          return PW_err   

     ## cal the goal functon

     def Goal_Function(q_cont,sta_Y,sta_Theta):
        Jh=1/2*Integrate((sta_Y-ex_Yd)**2,mesh_Auxi, order=3)+1/2*Integrate((sta_Theta-ex_ThetaD)**2, mesh, order=3)+alpha/2*Integrate(q_cont**2, mesh_Auxi, definedon=mesh_Auxi.Boundaries('outer'), order=3)
        return Jh

     #Sum_Goal_Fun0=np.zeros(LenT-1)

     #for  k in range(LenT-1):
          #t.Set(T_time[k+1])
          ##Sum_Goal_Fun0[k]= 1/2*Integrate((ex_Yd)**2,mesh_Auxi, order=3)


     #print("Goal value of uncontrol:", DeltaT*Sum_Goal_Fun0.sum())

     Auxi_Fun=GridFunction(PHeat_Sp)
     Auxi_Fun_HH=GridFunction(PHeat_Sp)

     Auxi_Y=GridFunction(Vel_Sp)
     Auxi_Theta=GridFunction(PHeat_Sp)

     for i in range(MaxIter):
          state_NSHEAT(mesh,t,T_time,DeltaT, Init_Y, Init_Theta, R_ns, R_heat, u_a, u_b, alpha, Sol_Kappa, Sol_Y, Sol_Theta)

          Sol_Mu=GridFunction(Vel_Sp,multidim=0)
          Sol_Kappa=GridFunction(PHeat_Sp,multidim=0)

          adjoint_NSHEAT(mesh, t, T_time, DeltaT, ex_Yd,ex_ThetaD,  Sol_Y, Sol_Theta, Sol_Mu, Sol_Kappa)
          
          Sum_Q=np.zeros(LenT-1)
     # Sum_Goal_Fun=np.zeros(LenT-1)
          
          for  k in range(LenT-1):
               t.Set(T_time[k+1])

               Auxi_Fun.vec.data=-eta/alpha*Sol_Kappa.vecs[LenT-k-2]
               Cont_n=IfPos(Auxi_Fun-u_b,u_b,0)+IfPos(Auxi_Fun-u_a,0,u_a)+IfPos((Auxi_Fun-u_b)*(Auxi_Fun-u_a),0,Auxi_Fun)
               
               Auxi_Fun_HH.vec.data=-eta/alpha*Auxi_Sol_Kappa.vecs[LenT-k-2]
               Cont_Nminus1=IfPos(Auxi_Fun_HH-u_b,u_b,0)+IfPos(Auxi_Fun_HH-u_a,0,u_a)+IfPos((Auxi_Fun_HH-u_b)*(Auxi_Fun_HH-u_a),0,Auxi_Fun_HH)
               Sum_Q[k]=Cal_err(Cont_Nminus1,Cont_n)

               Auxi_Y.vec.data=Sol_Y.vecs[k+1]
               Auxi_Theta.vec.data=Sol_Theta.vecs[k+1]

               #Sum_Goal_Fun[k]=Goal_Function(Cont_n,Auxi_Y,Auxi_Theta)

          IterErr=sqrt(DeltaT*Sum_Q.sum())
          #Obj_Fun=DeltaT*Sum_Goal_Fun.sum()

          print("Iter err: ",IterErr)
          #print("The value of the cost function", Obj_Fun)

          if IterErr<1e-6:
               break
          
          for j in range(LenT-1):
              Auxi_Sol_Kappa.vecs[j]=Sol_Kappa.vecs[j]


     return Sol_Y, Sol_Theta, Sol_Mu, Sol_Kappa
