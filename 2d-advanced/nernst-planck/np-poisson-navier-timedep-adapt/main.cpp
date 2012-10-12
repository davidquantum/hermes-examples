#define HERMES_REPORT_ALL
#define HERMES_REPORT_FILE "application.log"
#include "definitions.h"

#include "timestep_controller.h"

/** \addtogroup e_newton_np_timedep_adapt_system Newton Time-dependant System with Adaptivity
 \{
 \brief This example shows how to combine the automatic adaptivity with the Newton's method for a nonlinear time-dependent PDE system.

 This example shows how to combine the automatic adaptivity with the
 Newton's method for a nonlinear time-dependent PDE system.
 The time discretization is done using implicit Euler or
 Crank Nicholson method (see parameter TIME_DISCR).
 The following PDE's are solved:
 Nernst-Planck (describes the diffusion and migration of charged particles):
 \f[dC/dt - D*div[grad(C)] - K*C*div[grad(\phi)]=0,\f]
 where D and K are constants and C is the cation concentration variable,
 phi is the voltage variable in the Poisson equation:
 \f[ - div[grad(\phi)] = L*(C - C_0),\f]
 where \f$C_0\f$, and L are constant (anion concentration). \f$C_0\f$ is constant
 anion concentration in the domain and L is material parameter.
 So, the equation variables are phi and C and the system describes the
 migration/diffusion of charged particles due to applied voltage.
 The simulation domain looks as follows:
 \verbatim
      Top
     +----------+
     |          |
 Side|          |Side
     |          |
     +----------+
      Bottom
 \endverbatim
 For the Nernst-Planck equation, all the boundaries are natural i.e. Neumann.
 Which basically means that the normal derivative is 0:
 \f[ BC: -D*dC/dn - K*C*d\phi/dn = 0 \f]
 For Poisson equation, boundary 1 has a natural boundary condition
 (electric field derivative is 0).
 The voltage is applied to the boundaries 2 and 3 (Dirichlet boundaries)
 It is possible to adjust system paramter VOLT_BOUNDARY to apply
 Neumann boundary condition to 2 (instead of Dirichlet). But by default:
  - BC 2: \f$\phi = VOLTAGE\f$
  - BC 3: \f$\phi = 0\f$
  - BC 1: \f$\frac{d\phi}{dn} = 0\f$
 */

// Parameters to tweak the amount of output to the console.
#define NOSCREENSHOT

// True if scaled dimensionless variables are used, false otherwise.
bool SCALED = true;  

/*** Fundamental coefficients ***/

// [m^2/s] Diffusion coefficient.
const double D = 10e-11; 	                        
// [J/mol*K] Gas constant.
const double R = 8.31; 		                        
// [K] Aboslute temperature.
const double T = 293; 		                        
// [s * A / mol] Faraday constant.
const double F = 96485.3415;	                    
// [F/m] Electric permeability.
const double eps = 2.5e-2; 	                      
// Mobility of ions.
const double mu = D / (R * T);                    
// Charge number.
const double z = 1;		                            
// Constant for equation.
const double K = z * mu * F;                      
// Constant for equation.
const double L =  F / eps;	                      
// [mol/m^3] Anion and counterion concentration.
const double C0 = 1200;	                          

// Scaling constants.
// Scaling const, domain thickness [m].
const double l = 200e-6;                  
// Debye length [m].
double lambda = Hermes::sqrt((eps)*R*T/(2.0*F*F*C0)); 
double epsilon = lambda/l;

// mechanical parameters
const double mech_E = 0.5e9;                      // [Pa]
const double mech_nu = 0.487;                     // Poisson ratio
const double mech_mu = mech_E / (2 * (1 + mech_nu));
const double mech_lambda = mech_E * mech_nu / ((1 + mech_nu) * (1 - 2 * mech_nu));
const double lin_force_coup = 1e5;


// [V] Applied voltage.
const double VOLTAGE = 1;                         
const double SCALED_VOLTAGE = VOLTAGE*F/(R*T);

/* Simulation parameters */

const double T_FINAL = 3;
double INIT_TAU = 0.05;
// Size of the time step.
double *TAU = &INIT_TAU;                          

// Scaling time variables.
//double SCALED_INIT_TAU = INIT_TAU*D/(lambda * l);
//double TIME_SCALING = lambda * l / D;

// Initial polynomial degree of all mesh elements.
const int P_INIT = 2;       	                    
// Number of initial refinements.
const int REF_INIT = 2;     	                    
// Multimesh?
const bool MULTIMESH = true;	                    
// 1 for implicit Euler, 2 for Crank-Nicolson.
const int TIME_DISCR = 2;                         

// Stopping criterion for Newton on coarse mesh.
const double NEWTON_TOL_COARSE = 0.01;            
// Stopping criterion for Newton on fine mesh.
const double NEWTON_TOL_FINE = 0.05;              
// Maximum allowed number of Newton iterations.
const int NEWTON_MAX_ITER = 100;                  

// Every UNREF_FREQth time step the mesh is unrefined.
const int UNREF_FREQ = 1;                         
// This is a quantitative parameter of the adapt(...) function and
// it has different meanings for various adaptive strategies.
const double THRESHOLD = 0.3;                     
// Adaptive strategy:
// STRATEGY = 0 ... refine elements until sqrt(THRESHOLD) times total
//   error is processed. If more elements have similar errors, refine
//   all to keep the mesh symmetric.
// STRATEGY = 1 ... refine all elements whose error is larger
//   than THRESHOLD times maximum element error.
// STRATEGY = 2 ... refine all elements whose error is larger
//   than THRESHOLD.
const int STRATEGY = 0;                           
// Predefined list of element refinement candidates. Possible values are
// H2D_P_ISO, H2D_P_ANISO, H2D_H_ISO, H2D_H_ANISO, H2D_HP_ISO,
// H2D_HP_ANISO_H, H2D_HP_ANISO_P, H2D_HP_ANISO.
const CandList CAND_LIST = H2D_HP_ANISO;          
// Maximum allowed level of hanging nodes:
// MESH_REGULARITY = -1 ... arbitrary level hangning nodes (default),
// MESH_REGULARITY = 1 ... at most one-level hanging nodes,
// MESH_REGULARITY = 2 ... at most two-level hanging nodes, etc.
// Note that regular meshes are not supported, this is due to
// their notoriously bad performance.
const int MESH_REGULARITY = -1;                   
// This parameter influences the selection of
// candidates in hp-adaptivity. Default value is 1.0. 
const double CONV_EXP = 1.0;                      
// To prevent adaptivity from going on forever.
const int NDOF_STOP = 5000;	                      
// Stopping criterion for adaptivity.
const double ERR_STOP = 2;                      
// Matrix solver: SOLVER_AMESOS, SOLVER_AZTECOO, SOLVER_MUMPS,
// SOLVER_PETSC, SOLVER_SUPERLU, SOLVER_UMFPACK.
MatrixSolverType matrix_solver = SOLVER_UMFPACK;  

// Weak forms.
#include "definitions.cpp"

// Boundary markers.
const std::string BDY_SIDE_FIX = "Sidefix";
const std::string BDY_TOP = "Top";
const std::string BDY_BOT = "Bottom";

// scaling methods

double scaleTime(double t) {
  return SCALED ?  t * D / (lambda * l) : t;
}

double scaleVoltage(double phi) {
  return SCALED ? phi * F / (R * T) : phi;
}

double scaleConc(double C) {
  return SCALED ? C / C0 : C;
}

double physTime(double t) {
  return SCALED ? lambda * l * t / D : t;
}

double physConc(double C) {
  return SCALED ? C0 * C : C;
}

double physVoltage(double phi) {
  return SCALED ? phi * R * T / F : phi;
}

double disp_boundary() {
  return 0.0;
}

double SCALED_INIT_TAU = scaleTime(INIT_TAU);



int main (int argc, char* argv[]) {


  // Load the mesh file.
  Mesh C_mesh, phi_mesh, U_mesh, basemesh;
  MeshReaderH2D mloader;
  mloader.load("small.mesh", &basemesh);
  
  if (SCALED) {
    bool ret = basemesh.rescale(l, l);
    if (ret) {
      info("SCALED mesh is used");
    } else {
      info("UNSCALED mesh is used");
    }
  }

  // When nonadaptive solution, refine the mesh.
  basemesh.refine_towards_boundary(BDY_TOP, REF_INIT);
  basemesh.refine_towards_boundary(BDY_BOT, REF_INIT - 1);
  basemesh.refine_all_elements(1);
  basemesh.refine_all_elements(1);
  C_mesh.copy(&basemesh);
  phi_mesh.copy(&basemesh);
  U_mesh.copy(&basemesh);

  DefaultEssentialBCConst<double> bc_phi_voltage(BDY_TOP, scaleVoltage(VOLTAGE));
  DefaultEssentialBCConst<double> bc_phi_zero(BDY_BOT, scaleVoltage(0.0));
  DefaultEssentialBCConst<double> bc_U1_zero(BDY_SIDE_FIX, disp_boundary());
  DefaultEssentialBCConst<double> bc_U2_zero(BDY_SIDE_FIX, disp_boundary());

  EssentialBCs<double> bcs_phi(
      Hermes::vector<EssentialBoundaryCondition<double>* >(&bc_phi_voltage, &bc_phi_zero));
  EssentialBCs<double> bcs_U1(&bc_U1_zero);
  EssentialBCs<double> bcs_U2(&bc_U2_zero);

  //EssentialBCs<double> bcs_U1(
  //    Hermes::vector<EssentialBoundaryCondition<double>* >(&bc_U1_zero));
  //EssentialBCs<double> bcs_U2(
  //    Hermes::vector<EssentialBoundaryCondition<double>* >(&bc_U2_zero));

  // Spaces for concentration and the voltage.
  H1Space<double> C_space(&C_mesh, P_INIT);
  H1Space<double> phi_space(MULTIMESH ? &phi_mesh : &C_mesh, &bcs_phi, P_INIT);
  H1Space<double> U1_space(MULTIMESH ? & U_mesh : &C_mesh, &bcs_U1, P_INIT);
  H1Space<double> U2_space(MULTIMESH ? & U_mesh : &C_mesh, &bcs_U2, P_INIT);

  Solution<double> C_sln, C_ref_sln;
  Solution<double> phi_sln, phi_ref_sln;
  Solution<double> U1_sln, U1_ref_sln;
  Solution<double> U2_sln, U2_ref_sln;

  // Assign initial condition to mesh.
  ConstantSolution<double> C_prev_time(&C_mesh, scaleConc(C0));
  ConstantSolution<double> phi_prev_time(MULTIMESH ? &phi_mesh : &C_mesh, 0.0);
  ConstantSolution<double> U1_prev_time(MULTIMESH ? &U_mesh : &C_mesh, 0.0);
  ConstantSolution<double> U2_prev_time(MULTIMESH ? &U_mesh : &C_mesh, 0.0);

  // XXX not necessary probably
  if (SCALED) {
    TAU = &SCALED_INIT_TAU;
  }

  // The weak form for 2 equations.
  WeakForm<double> *wf;
  if (TIME_DISCR == 2) {
    if (SCALED) {
      wf = new ScaledWeakFormPNPEulerCranic(TAU, epsilon, C0 * lin_force_coup,
        mech_mu, mech_lambda, l, &C_prev_time, &phi_prev_time);
      info("Scaled weak form, with time step %g and epsilon %g", *TAU, epsilon);
    } else {
      error("Non-scaled form has not been implemented yet");
    }
  } else {
    error("Forward Euler is not implemented yet");
  }

  DiscreteProblem<double> dp_coarse(wf, Hermes::vector<const Space<double> *>(&C_space, 
      &phi_space, &U1_space, &U2_space));

  NewtonSolver<double>* solver_coarse = new NewtonSolver<double>(&dp_coarse, matrix_solver);

  // Project the initial condition on the FE space to obtain initial
  // coefficient vector for the Newton's method.
  info("Projecting to obtain initial vector for the Newton's method.");
  int ndof = Space<double>::get_num_dofs(Hermes::vector<const Space<double>*>(&C_space, 
      &phi_space, &U1_space, &U2_space));
  double* coeff_vec_coarse = new double[ndof] ;

  OGProjection<double>::project_global(Hermes::vector<const Space<double> *>(
          &C_space, &phi_space, &U1_space, &U2_space),
      Hermes::vector<MeshFunction<double> *>(&C_prev_time, &phi_prev_time,
          &U1_prev_time, &U2_prev_time), coeff_vec_coarse, matrix_solver);

  // Create a selector which will select optimal candidate.
  H1ProjBasedSelector<double> selector(CAND_LIST, CONV_EXP, H2DRS_DEFAULT_ORDER);

  // Visualization windows.
  char title[1000];
  ScalarView Cview("Concentration [mol/m3]", new WinGeom(0, 0, 800, 800));
  ScalarView phiview("Voltage [V]", new WinGeom(650, 0, 600, 600));
  ScalarView U1view("X-directional displacement", new WinGeom(10, 10, 800, 800));
  ScalarView U2view("Y-directional displacement", new WinGeom(660, 10, 600, 600));
  OrderView Cordview("C order", new WinGeom(0, 300, 600, 600));
  OrderView phiordview("Phi order", new WinGeom(600, 300, 600, 600));

  Cview.show(&C_prev_time);
  Cordview.show(&C_space);
  phiview.show(&phi_prev_time);
  phiordview.show(&phi_space);
  U1view.show(&U1_prev_time);
  U2view.show(&U2_prev_time);

  // Newton's loop on the coarse mesh.
  info("Solving initial coarse mesh");
  try
  {
    solver_coarse->solve(coeff_vec_coarse, NEWTON_TOL_COARSE, NEWTON_MAX_ITER);
  }
  catch(Hermes::Exceptions::Exception e)
  {
    e.printMsg();
    error("Newton's iteration failed.");
  };

  //View::wait(HERMES_WAIT_KEYPRESS);

  // Translate the resulting coefficient vector into the Solution<double> sln.
  Solution<double>::vector_to_solutions(solver_coarse->get_sln_vector(), 
      Hermes::vector<const Space<double> *>(&C_space, &phi_space, &U1_space, &U2_space),
      Hermes::vector<Solution<double> *>(&C_sln, &phi_sln, &U1_sln, &U2_sln));

  Cview.show(&C_sln);
  phiview.show(&phi_sln);
  U1view.show(&U1_sln);
  U2view.show(&U2_sln);

  // Cleanup after the Newton loop on the coarse mesh.
  delete solver_coarse;
  delete[] coeff_vec_coarse;
  
  // Time stepping loop.
  PidTimestepController pid(scaleTime(T_FINAL), true, scaleTime(INIT_TAU));
  TAU = pid.timestep;
  info("Starting time iteration with the step %g", *TAU);


  do {
    pid.begin_step();
    // Periodic global derefinements.
    if (pid.get_timestep_number() > 1 && pid.get_timestep_number() % UNREF_FREQ == 0)
    {
      info("Global mesh derefinement.");
      C_mesh.copy(&basemesh);
      if (MULTIMESH)
      {
        phi_mesh.copy(&basemesh);
        U_mesh.copy(&basemesh);
      }
      C_space.set_uniform_order(P_INIT);
      phi_space.set_uniform_order(P_INIT);
      U1_space.set_uniform_order(P_INIT);
      U2_space.set_uniform_order(P_INIT);

    }
  // TODO from here

    // Adaptivity loop. Note: C_prev_time and Phi_prev_time must not be changed during spatial adaptivity.
    bool done = false; int as = 1;
    double err_est;
    do {
      info("Time step %d, adaptivity step %d:", pid.get_timestep_number(), as);

      // Construct globally refined reference mesh
      // and setup reference space.
      Hermes::vector<Space<double> *>* ref_spaces =
          Space<double>::construct_refined_spaces(Hermes::vector<Space<double> *>(
            &C_space, &phi_space, &U1_space, &U2_space));
      Hermes::vector<const Space<double>*> ref_spaces_const((*ref_spaces)[0], (*ref_spaces)[1],
          (*ref_spaces)[2], (*ref_spaces)[3]);

      DiscreteProblem<double>* dp = new DiscreteProblem<double>(wf, ref_spaces_const);
      int ndof_ref = Space<double>::get_num_dofs(ref_spaces_const);

      double* coeff_vec = new double[ndof_ref];

      NewtonSolver<double>* solver = new NewtonSolver<double>(dp, matrix_solver);

      // Calculate initial coefficient vector for Newton on the fine mesh.
      if (as == 1 && pid.get_timestep_number() == 1) {
        info("Projecting coarse mesh solution to obtain coefficient vector on new fine mesh.");
        OGProjection<double>::project_global(ref_spaces_const,
              Hermes::vector<MeshFunction<double> *>(&C_sln, &phi_sln, &U1_sln, &U2_sln),
              coeff_vec, matrix_solver);
      }
      else {
        info("Projecting previous fine mesh solution to obtain coefficient vector on new fine mesh.");
        OGProjection<double>::project_global(ref_spaces_const,
           Hermes::vector<MeshFunction<double> *>(&C_ref_sln, &phi_ref_sln, &U1_ref_sln, &U2_ref_sln),
           coeff_vec, matrix_solver);
      }
      if (as > 1) {
        // Now deallocate the previous mesh
        info("Delallocating the previous mesh");
        delete C_ref_sln.get_mesh();
        delete phi_ref_sln.get_mesh();
        delete U1_ref_sln.get_mesh();
        delete U2_ref_sln.get_mesh();
      }

      // Newton's loop on the fine mesh.
      info("Solving on fine mesh:");
       try
      {
        solver->solve(coeff_vec, NEWTON_TOL_FINE, NEWTON_MAX_ITER);
      }
      catch(Hermes::Exceptions::Exception e)
      {
        e.printMsg();
        error("Newton's iteration failed.");
      };

      // Store the result in ref_sln.
      Solution<double>::vector_to_solutions(solver->get_sln_vector(), ref_spaces_const,
          Hermes::vector<Solution<double> *>(&C_ref_sln, &phi_ref_sln, &U1_ref_sln, &U2_ref_sln));

      // Projecting reference solution onto the coarse mesh
      info("Projecting fine mesh solution on coarse mesh.");
      OGProjection<double>::project_global(Hermes::vector<const Space<double> *>(
            &C_space, &phi_space, &U1_space, &U2_space),
          Hermes::vector<Solution<double> *>(&C_ref_sln, &phi_ref_sln, &U1_ref_sln, &U2_ref_sln),
          Hermes::vector<Solution<double> *>(&C_sln, &phi_sln, &U1_sln, &U2_sln),
          matrix_solver);

      // Calculate element errors and total error estimate.
      info("Calculating error estimate.");
      Adapt<double>* adaptivity = new Adapt<double>(Hermes::vector<Space<double> *>(
            &C_space, &phi_space, &U1_space, &U2_space));
      Hermes::vector<double> err_est_rel;
      double err_est_rel_total = adaptivity->calc_err_est(Hermes::vector<Solution<double> *>(
            &C_sln, &phi_sln, &U1_sln, &U2_sln),
          Hermes::vector<Solution<double> *>(
            &C_ref_sln, &phi_ref_sln, &U1_ref_sln, &U2_ref_sln), &err_est_rel) * 100;

      // Report results.
      info("ndof_coarse[0]: %d, ndof_fine[0]: %d",
           C_space.get_num_dofs(), (*ref_spaces)[0]->get_num_dofs());
      info("err_est_rel[0]: %g%%", err_est_rel[0]*100);
      info("ndof_coarse[1]: %d, ndof_fine[1]: %d",
           phi_space.get_num_dofs(), (*ref_spaces)[1]->get_num_dofs());
      info("err_est_rel[1]: %g%%", err_est_rel[1]*100);
      info("ndof_coarse[2]: %d, ndof_fine[2]: %d",
           U1_space.get_num_dofs(), (*ref_spaces)[2]->get_num_dofs());
      info("err_est_rel[2]: %g%%", err_est_rel[2]*100);
      info("ndof_coarse[3]: %d, ndof_fine[3]: %d",
           U2_space.get_num_dofs(), (*ref_spaces)[3]->get_num_dofs());
      info("err_est_rel[3]: %g%%", err_est_rel[3]*100);
      // Report results.
      info("ndof_coarse_total: %d, ndof_fine_total: %d, err_est_rel: %g%%", 
         Space<double>::get_num_dofs(Hermes::vector<const Space<double> *>(&C_space, &phi_space, &U1_space, &U2_space)),
           Space<double>::get_num_dofs(ref_spaces_const), err_est_rel_total);

      // If err_est too large, adapt the mesh.
      if (err_est_rel_total < ERR_STOP) done = true;
      else 
      {
        info("Adapting the coarse mesh.");
        done = adaptivity->adapt(Hermes::vector<Selector<double> *>(&selector, &selector, &selector, &selector),
          THRESHOLD, STRATEGY, MESH_REGULARITY);
        
        info("Adapted...");

        if (Space<double>::get_num_dofs(Hermes::vector<const Space<double> *>(&C_space, &phi_space, 
              &U1_space, &U2_space)) >= NDOF_STOP)
          done = true;
        else as++;
      }

      // Visualize the solution and mesh.
      info("Visualization procedures: C");
      char title[100];
      sprintf(title, "Solution[C], step# %d, step size %g, time %g, phys time %g",
          pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      Cview.set_title(title);
      Cview.show(&C_ref_sln);
      sprintf(title, "Mesh[C], step# %d, step size %g, time %g, phys time %g",
          pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      Cordview.set_title(title);
      Cordview.show(&C_space);
      
      info("Visualization procedures: phi");
      sprintf(title, "Solution[phi], step# %d, step size %g, time %g, phys time %g",
          pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      phiview.set_title(title);
      phiview.show(&phi_ref_sln);
      sprintf(title, "Mesh[phi], step# %d, step size %g, time %g, phys time %g",
          pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      phiordview.set_title(title);
      phiordview.show(&phi_space);

      info("Visualization procedures: U1");
      sprintf(title, "Solution[U1], step# %d, step size %g, time %g, phys time %g",
               pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      U1view.set_title(title);
      U1view.show(&U1_ref_sln);

      info("Visualization procedures: U2");
      sprintf(title, "Solution[U2], step# %d, step size %g, time %g, phys time %g",
               pid.get_timestep_number(), *TAU, pid.get_time(), physTime(pid.get_time()));
      U2view.set_title(title);
      U2view.show(&U2_ref_sln);


      //View::wait(HERMES_WAIT_KEYPRESS);

      // Clean up.
      delete solver;
      delete adaptivity;
      delete ref_spaces;
      delete dp;
      delete [] coeff_vec;
    }
    while (done == false);

    pid.end_step(Hermes::vector<Solution<double>*> (&C_ref_sln, &phi_ref_sln, &U1_ref_sln, &U2_ref_sln),
        Hermes::vector<Solution<double>*> (&C_prev_time, &phi_prev_time, &U1_prev_time, &U2_prev_time));
    // TODO! Time step reduction when necessary.

    // Copy last reference solution into sln_prev_time.
    C_prev_time.copy(&C_ref_sln);
    phi_prev_time.copy(&phi_ref_sln);
    U1_prev_time.copy(&U1_ref_sln);
    U2_prev_time.copy(&U2_ref_sln);

  } while (pid.has_next());

  // Wait for all views to be closed.
  View::wait();
  return 0;
}

