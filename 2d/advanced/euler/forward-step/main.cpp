#define HERMES_REPORT_INFO
#define HERMES_REPORT_FILE "application.log"
#include "hermes2d.h"

using namespace Hermes;
using namespace Hermes::Hermes2D;
using namespace Hermes::Hermes2D::Views;

// This example solves the compressible Euler equations using a basic
// piecewise-constant finite volume method.
//
// Equations: Compressible Euler equations, perfect gas state equation.
//
// Domain: forward facing step, see mesh file ffs.mesh
//
// BC: Normal velocity component is zero on solid walls.
//     Full supersonic state prescribed at inlet.
//     Pressure given at outlet, but used only if outlet flow is subsonic/
//
// IC: Constant supersonic state identical to inlet. 
//
// The following parameters can be changed:
// Visualization.
const bool HERMES_VISUALIZATION = true;           // Set to "true" to enable Hermes OpenGL visualization. 
const bool VTK_VISUALIZATION = false;              // Set to "true" to enable VTK output.
const unsigned int EVERY_NTH_STEP = 1;            // Set visual output for every nth step.

// Shock capturing.
bool SHOCK_CAPTURING = false;
// Quantitative parameter of the discontinuity detector.
double DISCONTINUITY_DETECTOR_PARAM = 1.0;

const int P_INIT = 0;                                   // Initial polynomial degree.                      
const int INIT_REF_NUM = 1;                             // Number of initial uniform mesh refinements.                       
double CFL_NUMBER = 1.0;                                // CFL value.
double time_step = 1E-4;                                // Initial time step.
const MatrixSolverType matrix_solver_type = SOLVER_UMFPACK;  // Possibilities: SOLVER_AMESOS, SOLVER_AZTECOO, SOLVER_MUMPS,
// SOLVER_PETSC, SOLVER_SUPERLU, SOLVER_UMFPACK.

// Equation parameters.
const double P_EXT = 1.0;         // Exterior pressure (dimensionless).
const double RHO_EXT = 1.4;       // Inlet density (dimensionless).   
const double V1_EXT = 3.0;        // Inlet x-velocity (dimensionless).
const double V2_EXT = 0.0;        // Inlet y-velocity (dimensionless).
const double KAPPA = 1.4;         // Kappa.

// Boundary markers.
const std::string BDY_SOLID_WALL_BOTTOM = "1";
const std::string BDY_OUTLET = "2";
const std::string BDY_SOLID_WALL_TOP = "3";
const std::string BDY_INLET = "4";

// Weak forms.
#include "../forms_explicit.cpp"

// Initial condition.
#include "../initial_condition.cpp"

int main(int argc, char* argv[])
{
  // Load the mesh.
  Mesh mesh;
  MeshReaderH2D mloader;
  mloader.load("ffs.mesh", &mesh);

  // Perform initial mesh refinements.
  for (int i = 0; i < INIT_REF_NUM; i++) 
    mesh.refine_all_elements(0);
  
  // Initialize boundary condition types and spaces with default shapesets.
  L2Space<double> space_rho(&mesh, P_INIT);
  L2Space<double> space_rho_v_x(&mesh, P_INIT);
  L2Space<double> space_rho_v_y(&mesh, P_INIT);
  L2Space<double> space_e(&mesh, P_INIT);
  int ndof = Space<double>::get_num_dofs(Hermes::vector<Space<double>*>(&space_rho, &space_rho_v_x, &space_rho_v_y, &space_e));
  info("ndof: %d", ndof);

  // Initialize solutions, set initial conditions.
  InitialSolutionEulerDensity prev_rho(&mesh, RHO_EXT);
  InitialSolutionEulerDensityVelX prev_rho_v_x(&mesh, RHO_EXT * V1_EXT);
  InitialSolutionEulerDensityVelY prev_rho_v_y(&mesh, RHO_EXT * V2_EXT);
  InitialSolutionEulerDensityEnergy prev_e(&mesh, QuantityCalculator::calc_energy(RHO_EXT, RHO_EXT * V1_EXT, RHO_EXT * V2_EXT, P_EXT, KAPPA));

  // Numerical flux.
  OsherSolomonNumericalFlux num_flux(KAPPA); 

  // Initialize weak formulation.
  EulerEquationsWeakFormSemiImplicitMultiComponent wf(&num_flux, KAPPA, RHO_EXT, V1_EXT, V2_EXT, P_EXT, BDY_SOLID_WALL_BOTTOM, BDY_SOLID_WALL_TOP, 
    BDY_INLET, BDY_OUTLET, &prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e, (P_INIT == 0));

  // Initialize the FE problem.
  DiscreteProblem<double> dp(&wf, Hermes::vector<Space<double>*>(&space_rho, &space_rho_v_x, &space_rho_v_y, &space_e));

  // If the FE problem is in fact a FV problem.
  if(P_INIT == 0)
    dp.set_fvm();  

  // Filters for visualization of Mach number, pressure and entropy.
  MachNumberFilter Mach_number(Hermes::vector<MeshFunction<double>*>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e), KAPPA);
  PressureFilter pressure(Hermes::vector<MeshFunction<double>*>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e), KAPPA);
  EntropyFilter entropy(Hermes::vector<MeshFunction<double>*>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e), KAPPA, RHO_EXT, P_EXT);

  ScalarView<double> pressure_view("Pressure", new WinGeom(0, 0, 600, 300));
  ScalarView<double> Mach_number_view("Mach number", new WinGeom(700, 0, 600, 300));
  ScalarView<double> entropy_production_view("Entropy estimate", new WinGeom(0, 400, 600, 300));

  // Set up the solver, matrix, and rhs according to the solver selection.
  SparseMatrix<double>* matrix = create_matrix<double>(matrix_solver_type);
  Vector<double>* rhs = create_vector<double>(matrix_solver_type);
  LinearSolver<double>* solver = create_linear_solver<double>(matrix_solver_type, matrix, rhs);

  // Set up CFL calculation class.
  CFLCalculation CFL(CFL_NUMBER, KAPPA);

  int iteration = 0; double t = 0;
  for(t = 0.0; t < 10.0; t += time_step)
  {
    info("---- Time step %d, time %3.5f.", iteration++, t);

    // Set the current time step.
    wf.set_time_step(time_step);

    // Assemble the stiffness matrix and rhs.
    info("Assembling the stiffness matrix and right-hand side vector.");
    dp.assemble(matrix, rhs);

    dp.get_last_profiling_output(std::cout);

    // Solve the matrix problem.
    info("Solving the matrix problem.");
    if(solver->solve())
      if(!SHOCK_CAPTURING)
      Solution<double>::vector_to_solutions(solver->get_sln_vector(), Hermes::vector<Space<double> *>(&space_rho, &space_rho_v_x, 
          &space_rho_v_y, &space_e), Hermes::vector<Solution<double> *>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e));
    else
        {      
          FluxLimiter flux_limiter(FluxLimiter::Kuzmin, solver->get_sln_vector(), Hermes::vector<Space<double> *>(&space_rho, &space_rho_v_x, 
            &space_rho_v_y, &space_e));

          flux_limiter.limit_second_orders_according_to_detector();

          flux_limiter.limit_according_to_detector();

          flux_limiter.get_limited_solutions(Hermes::vector<Solution<double> *>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e));
        }
    else
      error ("Matrix solver failed.\n");

    CFL.calculate_semi_implicit(Hermes::vector<Solution<double> *>(&prev_rho, &prev_rho_v_x, &prev_rho_v_y, &prev_e), &mesh, time_step);

        // Visualization.

    if((iteration - 1) % EVERY_NTH_STEP == 0) 
    {
      // Hermes visualization.
      if(HERMES_VISUALIZATION) 
      {
        Mach_number.reinit();
        pressure.reinit();
        entropy.reinit();
        pressure_view.show(&pressure);
        entropy_production_view.show(&entropy);
        Mach_number_view.show(&Mach_number);
      }
      // Output solution in VTK format.
      if(VTK_VISUALIZATION) 
      {
        pressure.reinit();
        Mach_number.reinit();
        Linearizer<double> lin;
        char filename[40];
        sprintf(filename, "pressure-3D-%i.vtk", iteration - 1);
        lin.save_solution_vtk(&pressure, filename, "Pressure", true);
        sprintf(filename, "Mach number-3D-%i.vtk", iteration - 1);
        lin.save_solution_vtk(&Mach_number, filename, "MachNumber", true);
      }
    }
  }

  pressure_view.close();
  entropy_production_view.close();
  Mach_number_view.close();

  return 0;
}
