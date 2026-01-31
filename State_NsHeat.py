# origin the projection method
from ngsolve import *

def state_NSHEAT(mesh,t,T_time,DeltaT, Y0, Theta0, R_ns, R_heat, u_a, u_b, alpha, Sol_Kappa, Sol_Y, Sol_theta):
   nu=0.1
   g_vec=CF((-10,10))
   eta=1
   Vel_Sp=VectorH1(mesh,order=1,dirichlet='outer')
   Vel_Sp.SetOrder(TRIG,3)
   Vel_Sp.Update()

   PHeat_Sp=H1(mesh,order=1)

   fes_VelP=Vel_Sp*PHeat_Sp


   # the time size 
   LenT=len(T_time)

   # define the finite element with NS and heat
   (u, p), (v, q)=fes_VelP.TnT()

   theta, psi=PHeat_Sp.TnT()

   # Give the initial condition
   Y_init=GridFunction(Vel_Sp)
   Y_init.vec.data=Y0.vec.data

   Theta_init=GridFunction(PHeat_Sp)
   Theta_init.vec.data=Theta0.vec.data
   

   Auxi_Fun=GridFunction(PHeat_Sp)
   re_q_cont=IfPos(Auxi_Fun-u_b,u_b,0)+IfPos(Auxi_Fun-u_a,0,u_a)+IfPos((Auxi_Fun-u_b)*(Auxi_Fun-u_a),0,Auxi_Fun)
   
   a_NS=BilinearForm(fes_VelP)
   b_NS=LinearForm(fes_VelP)


   a_NS+=InnerProduct(u,v)*dx+nu*DeltaT*InnerProduct(Grad(u),Grad(v))*dx+1/2*DeltaT*InnerProduct(Grad(u)*Y_init,v)*dx-1/2*DeltaT*InnerProduct(Grad(v)*Y_init, u)*dx-DeltaT*p*div(v)*dx-div(u)*q*dx

   b_NS+=InnerProduct(Y_init,v)*dx-DeltaT*Theta_init*InnerProduct(g_vec, v)*dx+DeltaT*InnerProduct(R_ns,v)*dx


   a_Heat=BilinearForm(PHeat_Sp)
   b_Heat=LinearForm(PHeat_Sp)

   a_Heat+=theta*psi*dx+DeltaT*grad(theta)*grad(psi)*dx+DeltaT*1/2*InnerProduct(Y_init,grad(theta))*psi*dx-1/2*DeltaT*InnerProduct(Y_init, grad(psi))*theta*dx+eta*DeltaT*theta*psi*ds
   
   b_Heat+=Theta_init*psi*dx+DeltaT*R_heat*psi*dx+eta*DeltaT*re_q_cont*psi*ds(bonus_intorder = 3)

   Sol_Y.vecs[0]=Y_init.vec.data
   Sol_theta.vecs[0]=Theta_init.vec.data
   
   
   Sol_YP=GridFunction(fes_VelP)

   for i in range(LenT-1):
       Auxi_Fun.vec.data=-eta/alpha*Sol_Kappa.vecs[LenT-i-2]
       t.Set(T_time[i+1])
      
    # prediction equation to solve convect-diffusion equation
       a_NS.Assemble()
       b_NS.Assemble()
   
       Sol_YP.vec.data=a_NS.mat.Inverse(fes_VelP.FreeDofs(),inverse='pardiso')*b_NS.vec
       

       a_Heat.Assemble()
       b_Heat.Assemble()
       
       Theta_init.vec.data=a_Heat.mat.Inverse(inverse='pardiso')*b_Heat.vec
       
       Y_init.vec.data=Sol_YP.components[0].vec.data

       Sol_Y.vecs[i+1]=Y_init.vec.data
       Sol_theta.vecs[i+1]=Theta_init.vec.data
       




       
