#include "header.h"

#include "timestep_controller.h"

// Weak forms
#include "definitions.cpp"

// Initial conditions.
#include "initial_conditions.cpp"



int main (int argc, char* argv[]) {


  // Load the mesh file.
  Mesh C_mesh, phi_mesh, basemesh;
  H2DReader mloader;
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

  DefaultEssentialBCConst<double> bc_phi_voltage(BDY_TOP, scaleVoltage(VOLTAGE));
  DefaultEssentialBCConst<double> bc_phi_zero(BDY_BOT, scaleVoltage(0.0));

  EssentialBCs<double> bcs_phi(
      Hermes::vector<EssentialBoundaryCondition<double>* >(&bc_phi_voltage, &bc_phi_zero));

  // Spaces for concentration and the voltage.
  H1Space<double> C_space(&C_mesh, P_INIT);
  H1Space<double> phi_space(MULTIMESH ? &phi_mesh : &C_mesh, &bcs_phi, P_INIT);

  Solution<double> C_sln, C_ref_sln;
  Solution<double> phi_sln, phi_ref_sln;

  // Assign initial condition to mesh.
  InitialSolutionConcentration C_prev_time(&C_mesh, scaleConc(C0));
  InitialSolutionVoltage phi_prev_time(MULTIMESH ? &phi_mesh : &C_mesh);

  // XXX not necessary probably
  if (SCALED) {
    TAU = &SCALED_INIT_TAU;
  }

  // The weak form for 2 equations.
  WeakForm<double> *wf;
  if (TIME_DISCR == 2) {
    if (SCALED) {
      wf = new ScaledWeakFormPNPCranic(TAU, epsilon, &C_prev_time, &phi_prev_time);
      info("Scaled weak form, with time step %g and epsilon %g", *TAU, epsilon);
    } else {
      wf = new WeakFormPNPCranic(TAU, C0, K, L, D, &C_prev_time, &phi_prev_time);
    }
  } else {
    if (SCALED)
      error("Forward Euler is not implemented for scaled problem");
    wf = new WeakFormPNPEuler(TAU, C0, K, L, D, &C_prev_time);
  }

  DiscreteProblem<double> dp_coarse(wf, Hermes::vector<Space<double> *>(&C_space, &phi_space));

  NewtonSolver<double>* solver_coarse = new NewtonSolver<double>(&dp_coarse, matrix_solver);

  // Project the initial condition on the FE space to obtain initial
  // coefficient vector for the Newton's method.
  info("Projecting to obtain initial vector for the Newton's method.");
  int ndof = Space<double>::get_num_dofs(Hermes::vector<Space<double>*>(&C_space, &phi_space));
  double* coeff_vec_coarse = new double[ndof] ;
  memset(coeff_vec_coarse, 0, ndof * sizeof(double));

  OGProjection<double>::project_global(Hermes::vector<Space<double> *>(&C_space, &phi_space),
      Hermes::vector<MeshFunction<double> *>(&C_prev_time, &phi_prev_time),
      coeff_vec_coarse, matrix_solver);

  // Create a selector which will select optimal candidate.
  H1ProjBasedSelector<double> selector(CAND_LIST, CONV_EXP, H2DRS_DEFAULT_ORDER);

  // Visualization windows.
  char title[1000];
  ScalarView<double> Cview("Concentration [mol/m3]", new WinGeom(0, 0, 800, 800));
  ScalarView<double> phiview("Voltage [V]", new WinGeom(650, 0, 600, 600));
  OrderView<double> Cordview("C order", new WinGeom(0, 300, 600, 600));
  OrderView<double> phiordview("Phi order", new WinGeom(600, 300, 600, 600));

  Cview.show(&C_prev_time);
  Cordview.show(&C_space);
  phiview.show(&phi_prev_time);
  phiordview.show(&phi_space);

  // Newton's loop on the coarse mesh.

  info("Solving initial coarse mesh");
  if (!solver_coarse->solve(coeff_vec_coarse, NEWTON_TOL_COARSE, NEWTON_MAX_ITER))
    error("Newton's iteration failed.");

  //View::wait(HERMES_WAIT_KEYPRESS);

  // Translate the resulting coefficient vector into the Solution sln.
  Solution<double>::vector_to_solutions(coeff_vec_coarse, Hermes::vector<Space<double> *>(&C_space, &phi_space),
                                Hermes::vector<Solution<double> *>(&C_sln, &phi_sln));

  Cview.show(&C_sln);
  phiview.show(&phi_sln);

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
      }
      C_space.set_uniform_order(P_INIT);
      phi_space.set_uniform_order(P_INIT);

    }

    // Adaptivity loop. Note: C_prev_time and Phi_prev_time must not be changed during spatial adaptivity.
    bool done = false; int as = 1;
    double err_est;
    do {
      info("Time step %d, adaptivity step %d:", pid.get_timestep_number(), as);

      // Construct globally refined reference mesh
      // and setup reference space.
      Hermes::vector<Space<double> *>* ref_spaces =
          Space<double>::construct_refined_spaces(Hermes::vector<Space<double> *>(&C_space, &phi_space));

      DiscreteProblem<double>* dp = new DiscreteProblem<double>(wf, *ref_spaces);
      int ndof_ref = Space<double>::get_num_dofs(*ref_spaces);

      double* coeff_vec = new double[ndof_ref];
      memset(coeff_vec, 0, ndof_ref * sizeof(double));

      NewtonSolver<double>* solver = new NewtonSolver<double>(dp, matrix_solver);

      // Calculate initial coefficient vector for Newton on the fine mesh.
      if (as == 1 && pid.get_timestep_number() == 1) {
        info("Projecting coarse mesh solution to obtain coefficient vector on new fine mesh.");
        OGProjection<double>::project_global(*ref_spaces,
              Hermes::vector<MeshFunction<double> *>(&C_sln, &phi_sln),
              coeff_vec, matrix_solver);
      }
      else {
        info("Projecting previous fine mesh solution to obtain coefficient vector on new fine mesh.");
        OGProjection<double>::project_global(*ref_spaces,
              Hermes::vector<MeshFunction<double> *>(&C_ref_sln, &phi_ref_sln),
              coeff_vec, matrix_solver);
      }
      if (as > 1) {
        // Now deallocate the previous mesh
        info("Delallocating the previous mesh");
        delete C_ref_sln.get_mesh();
        delete phi_ref_sln.get_mesh();
      }

      // Newton's loop on the fine mesh.
      info("Solving on fine mesh:");
      if (!solver->solve(coeff_vec, NEWTON_TOL_FINE, NEWTON_MAX_ITER))
          error("Newton's iteration failed.");

      // Store the result in ref_sln.
      Solution<double>::vector_to_solutions(coeff_vec, *ref_spaces,
                                    Hermes::vector<Solution<double> *>(&C_ref_sln, &phi_ref_sln));

      // Projecting reference solution onto the coarse mesh
      info("Projecting fine mesh solution on coarse mesh.");
      OGProjection<double>::project_global(Hermes::vector<Space<double> *>(&C_space, &phi_space),
                                   Hermes::vector<Solution<double> *>(&C_ref_sln, &phi_ref_sln),
                                   Hermes::vector<Solution<double> *>(&C_sln, &phi_sln),
                                   matrix_solver);

      // Calculate element errors and total error estimate.
      info("Calculating error estimate.");
      Adapt<double>* adaptivity = new Adapt<double>(Hermes::vector<Space<double> *>(&C_space, &phi_space));
      Hermes::vector<double> err_est_rel;
      double err_est_rel_total = adaptivity->calc_err_est(Hermes::vector<Solution<double> *>(&C_sln, &phi_sln),
                                 Hermes::vector<Solution<double> *>(&C_ref_sln, &phi_ref_sln), &err_est_rel) * 100;

      // Report results.
      info("ndof_coarse[0]: %d, ndof_fine[0]: %d",
           C_space.get_num_dofs(), (*ref_spaces)[0]->get_num_dofs());
      info("err_est_rel[0]: %g%%", err_est_rel[0]*100);
      info("ndof_coarse[1]: %d, ndof_fine[1]: %d",
           phi_space.get_num_dofs(), (*ref_spaces)[1]->get_num_dofs());
      info("err_est_rel[1]: %g%%", err_est_rel[1]*100);
      // Report results.
      info("ndof_coarse_total: %d, ndof_fine_total: %d, err_est_rel: %g%%", 
           Space<double>::get_num_dofs(Hermes::vector<Space<double> *>(&C_space, &phi_space)),
               Space<double>::get_num_dofs(*ref_spaces), err_est_rel_total);

      // If err_est too large, adapt the mesh.
      if (err_est_rel_total < ERR_STOP) done = true;
      else 
      {
        info("Adapting the coarse mesh.");
        done = adaptivity->adapt(Hermes::vector<Selector<double> *>(&selector, &selector),
          THRESHOLD, STRATEGY, MESH_REGULARITY);
        
        info("Adapted...");

        if (Space<double>::get_num_dofs(Hermes::vector<Space<double> *>(&C_space, &phi_space)) >= NDOF_STOP)
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
      //View::wait(HERMES_WAIT_KEYPRESS);

      // Clean up.
      delete solver;
      delete adaptivity;
      delete ref_spaces;
      delete dp;
      delete[] coeff_vec;
    }
    while (done == false);

    pid.end_step(Hermes::vector<Solution<double>*> (&C_ref_sln, &phi_ref_sln),
        Hermes::vector<Solution<double>*> (&C_prev_time, &phi_prev_time));
    // TODO! Time step reduction when necessary.

    // Copy last reference solution into sln_prev_time.
    C_prev_time.copy(&C_ref_sln);
    phi_prev_time.copy(&phi_ref_sln);

  } while (pid.has_next());

  // Wait for all views to be closed.
  View::wait();
  return 0;
}

