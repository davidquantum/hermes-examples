#include "../header.h"

// Weak forms
#include "../definitions.cpp"

// Initial conditions.
#include "../initial_conditions.cpp"

double current_time = 0.0;
ButcherTableType butcher_table_type = Implicit_RK_1;

int main (int argc, char* argv[]) {

  ButcherTable bt(butcher_table_type);
  if (bt.is_explicit()) info("Using a %d-stage explicit R-K method.", bt.get_size());
  if (bt.is_diagonally_implicit()) info("Using a %d-stage diagonally implicit R-K method.", bt.get_size());
  if (bt.is_fully_implicit()) info("Using a %d-stage fully implicit R-K method.", bt.get_size());


  // Load the mesh file.
  Mesh C_mesh, phi_mesh, basemesh;
  H2DReader mloader;
  mloader.load("../small.mesh", &basemesh);

  bool ret = basemesh.rescale(l, l);

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

  Solution<double> C_sln, phi_sln;
  Solution<double> C_ref_sln, phi_ref_sln;

  // Assign initial condition to mesh.
  InitialSolutionConcentration C_prev_time(&C_mesh, scaleConc(C0));
  InitialSolutionVoltage phi_prev_time(MULTIMESH ? &phi_mesh : &C_mesh);


  WeakForm<double> *wf;
  wf = new ScaledWeakFormPNPRungeKutta(epsilon);

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

  int ts = 1;
  do {
    // Periodic global derefinements.
    if (ts > 1 && ts % UNREF_FREQ == 0)
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
      info("Time step %d, adaptivity step %d:", ts, as);

      // Construct globally refined reference mesh
      // and setup reference space.
      Hermes::vector<Space<double> *>* ref_spaces =
          Space<double>::construct_refined_spaces(Hermes::vector<Space<double> *>(&C_space, &phi_space));

      DiscreteProblem<double>* dp = new DiscreteProblem<double>(wf, *ref_spaces);

      RungeKutta<double> runge_kutta(dp, &bt, matrix_solver, false, true, Hermes::vector<int>(1));

      // Perform one Runge-Kutta time step according to the selected Butcher's table.
      info("Runge-Kutta time step (t = %g s, tau = %g s, stages: %d).",
           current_time, INIT_TAU, bt.get_size());
      bool freeze_jacobian = true;
      bool block_diagonal_jacobian = true;
      bool verbose = true;
      if (!runge_kutta.rk_time_step(scaleTime(current_time), scaleTime(INIT_TAU),
            Hermes::vector<Solution<double>*>(&C_prev_time, &phi_prev_time),
            Hermes::vector<Solution<double>*>(&C_ref_sln, &phi_ref_sln),
                                    freeze_jacobian, block_diagonal_jacobian,
                                    verbose, NEWTON_TOL_FINE, NEWTON_MAX_ITER)) {
        error("Runge-Kutta time step failed, try to decrease time step size.");
      }

      Solution<double> C_coarse, phi_coarse;

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
      sprintf(title, "Solution[C], step# %d, step size %g, time %g",
          ts, INIT_TAU, current_time, physTime(current_time));
      Cview.set_title(title);
      Cview.show(&C_ref_sln);
      sprintf(title, "Mesh[C], step# %d, step size %g, time %g",
          ts, INIT_TAU, current_time);
      Cordview.set_title(title);
      Cordview.show(&C_space);

      info("Visualization procedures: phi");
      sprintf(title, "Solution[phi], step# %d, step size %g, time %g",
          ts, INIT_TAU, current_time);
      phiview.set_title(title);
      phiview.show(&phi_ref_sln);
      sprintf(title, "Mesh[phi], step# %d, step size %g, time %g",
          ts, INIT_TAU, current_time);
      phiordview.set_title(title);
      phiordview.show(&phi_space);
      //View::wait(HERMES_WAIT_KEYPRESS);

      // Clean up.
      delete adaptivity;
      delete ref_spaces;
      delete dp;
      if (!done) {
        info("Delallocating the previous mesh");
        delete C_ref_sln.get_mesh();
        delete phi_ref_sln.get_mesh();
      }
    }
    while (done == false);


    // Copy last reference solution into sln_prev_time.
    C_prev_time.copy(&C_ref_sln);
    phi_prev_time.copy(&phi_ref_sln);
    current_time += INIT_TAU;
    ts++;

  } while (current_time < T_FINAL);

  // Wait for all views to be closed.
  View::wait();
  return 0;
}

