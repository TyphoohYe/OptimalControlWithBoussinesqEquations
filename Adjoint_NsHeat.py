from ngsolve import *

def adjoint_NSHEAT(mesh, t, T_time, DeltaT, YD, ex_ThetaD, Sol_Y, Sol_Theta, Sol_Mu, Sol_Kappa):
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
   (mu, p), (w, q)=fes_VelP.TnT()

   kappa, xi=PHeat_Sp.TnT()

   # Give the initial condition
   Y_nminus1=GridFunction(Vel_Sp)
   
   Y_nplus1=GridFunction(Vel_Sp)
   Theta_nplus1=GridFunction(PHeat_Sp)

   Y_n=GridFunction(Vel_Sp)
   Theta_n=GridFunction(PHeat_Sp)
 

   Sol_MuP=GridFunction(fes_VelP)

   Mu_nplus1=GridFunction(Vel_Sp)
   Kappa_nplus1=GridFunction(PHeat_Sp)
   
   a_NS=BilinearForm(fes_VelP)
   b_NS=LinearForm(fes_VelP)

   a_NS+=InnerProduct(mu,w)*dx+nu*DeltaT*InnerProduct(Grad(mu),Grad(w))*dx+1/2*DeltaT*InnerProduct(Grad(w)*Y_nminus1,mu)*dx-1/2*DeltaT*InnerProduct(Grad(mu)*Y_nminus1, w)*dx-DeltaT*p*div(w)*dx-div(mu)*q*dx

   b_NS+=InnerProduct(Mu_nplus1,w)*dx+DeltaT*InnerProduct(Y_n-YD,w)*dx(bonus_intorder = 3)\
   -1/2*DeltaT*InnerProduct(Grad(Y_nplus1)*w,Mu_nplus1)*dx+1/2*DeltaT*InnerProduct(Grad(Mu_nplus1)*w, Y_nplus1)*dx\
   -1/2*DeltaT*InnerProduct(w,grad(Theta_nplus1))*Kappa_nplus1*dx+1/2*DeltaT*InnerProduct(w, grad(Kappa_nplus1))*Theta_nplus1*dx


   a_Heat=BilinearForm(PHeat_Sp)
   b_Heat=LinearForm(PHeat_Sp)

   a_Heat+=kappa*xi*dx+DeltaT*grad(kappa)*grad(xi)*dx+DeltaT*1/2*InnerProduct(Y_nminus1,grad(xi))*kappa*dx-1/2*DeltaT*InnerProduct(Y_nminus1, grad(kappa))*xi*dx+eta*DeltaT*kappa*xi*ds
   
   b_Heat+=Kappa_nplus1*xi*dx-DeltaT*InnerProduct(Mu_nplus1, g_vec)*xi*dx\
        +DeltaT*(Theta_n-ex_ThetaD)*xi*dx
   
   rever_range=list(reversed(list(range(1,LenT))))

   for i in rever_range:
        Y_nminus1.vec.data=Sol_Y.vecs[i-1]
        
        Y_n.vec.data=Sol_Y.vecs[i]
        Theta_n.vec.data=Sol_Theta.vecs[i]
        
        t.Set(T_time[i-1])

        a_NS.Assemble()
        b_NS.Assemble()
        
        Sol_MuP.vec.data=a_NS.mat.Inverse(fes_VelP.FreeDofs())*b_NS.vec

        a_Heat.Assemble()
        b_Heat.Assemble()

        Kappa_nplus1.vec.data=a_Heat.mat.Inverse()*b_Heat.vec
        Sol_Kappa.AddMultiDimComponent(Kappa_nplus1.vec)

        Mu_nplus1.vec.data=Sol_MuP.components[0].vec.data
        Sol_Mu.AddMultiDimComponent(Mu_nplus1.vec)

        Y_nplus1.vec.data=Sol_Y.vecs[i]
        Theta_nplus1.vec.data=Sol_Theta.vecs[i]                       





        
        
        

        