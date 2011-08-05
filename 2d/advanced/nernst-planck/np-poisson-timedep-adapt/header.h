#define HERMES_REPORT_WARN
#define HERMES_REPORT_INFO
#define HERMES_REPORT_VERBOSE
#define HERMES_REPORT_FILE "application.log"

#include "hermes2d.h"

using namespace Hermes;
using namespace Hermes::Hermes2D;
using namespace Hermes::Hermes2D::Views;
using namespace RefinementSelectors;

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


bool SCALED = true;  // true if scaled dimensionless variables are used, false otherwise


/*** Fundamental coefficients ***/
const double D = 10e-11;                          // [m^2/s] Diffusion coefficient.
const double R = 8.31;                            // [J/mol*K] Gas constant.
const double T = 293;                             // [K] Aboslute temperature.
const double F = 96485.3415;                      // [s * A / mol] Faraday constant.
const double eps = 2.5e-2;                        // [F/m] Electric permeability.
const double mu = D / (R * T);                    // Mobility of ions.
const double z = 1;                               // Charge number.
const double K = z * mu * F;                      // Constant for equation.
const double L =  F / eps;                        // Constant for equation.
const double C0 = 1200;                           // [mol/m^3] Anion and counterion concentration.

// Scaling constants
const double l = 200e-6;                  // scaling const, domain thickness [m]
double lambda = Hermes::sqrt((eps)*R*T/(2.0*F*F*C0)); //Debye length [m]
double epsilon = lambda/l;

const double VOLTAGE = 1;                         // [V] Applied voltage.
const double SCALED_VOLTAGE = VOLTAGE*F/(R*T);



/* Simulation parameters */
const double T_FINAL = 3;
double INIT_TAU = 0.05;
double *TAU = &INIT_TAU;                          // Size of the time step

// scaling time variables
//double SCALED_INIT_TAU = INIT_TAU*D/(lambda * l);
//double TIME_SCALING = lambda * l / D;

const int P_INIT = 2;                             // Initial polynomial degree of all mesh elements.
const int REF_INIT = 3;                           // Number of initial refinements.
const bool MULTIMESH = true;                      // Multimesh?
const int TIME_DISCR = 2;                         // 1 for implicit Euler, 2 for Crank-Nicolson.

const double NEWTON_TOL_COARSE = 0.01;            // Stopping criterion for Newton on coarse mesh.
const double NEWTON_TOL_FINE = 0.05;              // Stopping criterion for Newton on fine mesh.
const int NEWTON_MAX_ITER = 100;                  // Maximum allowed number of Newton iterations.

const int UNREF_FREQ = 1;                         // every UNREF_FREQth time step the mesh is unrefined.
const double THRESHOLD = 0.3;                     // This is a quantitative parameter of the adapt(...) function and
                                                  // it has different meanings for various adaptive strategies (see below).
const int STRATEGY = 0;                           // Adaptive strategy:
                                                  // STRATEGY = 0 ... refine elements until sqrt(THRESHOLD) times total
                                                  //   error is processed. If more elements have similar errors, refine
                                                  //   all to keep the mesh symmetric.
                                                  // STRATEGY = 1 ... refine all elements whose error is larger
                                                  //   than THRESHOLD times maximum element error.
                                                  // STRATEGY = 2 ... refine all elements whose error is larger
                                                  //   than THRESHOLD.
                                                  // More adaptive strategies can be created in adapt_ortho_h1.cpp.
const CandList CAND_LIST = H2D_HP_ANISO;          // Predefined list of element refinement candidates. Possible values are
                                                  // H2D_P_ISO, H2D_P_ANISO, H2D_H_ISO, H2D_H_ANISO, H2D_HP_ISO,
                                                  // H2D_HP_ANISO_H, H2D_HP_ANISO_P, H2D_HP_ANISO.
                                                  // See User Documentation for details.
const int MESH_REGULARITY = -1;                   // Maximum allowed level of hanging nodes:
                                                  // MESH_REGULARITY = -1 ... arbitrary level hangning nodes (default),
                                                  // MESH_REGULARITY = 1 ... at most one-level hanging nodes,
                                                  // MESH_REGULARITY = 2 ... at most two-level hanging nodes, etc.
                                                  // Note that regular meshes are not supported, this is due to
                                                  // their notoriously bad performance.
const double CONV_EXP = 1.0;                      // Default value is 1.0. This parameter influences the selection of
                                                  // cancidates in hp-adaptivity. See get_optimal_refinement() for details.
const int NDOF_STOP = 5000;                       // To prevent adaptivity from going on forever.
const double ERR_STOP = 0.1;                      // Stopping criterion for adaptivity (rel. error tolerance between the
                                                  // fine mesh and coarse mesh solution in percent).
MatrixSolverType matrix_solver = SOLVER_UMFPACK;  // Possibilities: SOLVER_AMESOS, SOLVER_AZTECOO, SOLVER_MUMPS,
                                                  // SOLVER_PETSC, SOLVER_SUPERLU, SOLVER_UMFPACK.

// Boundary markers.
const std::string BDY_SIDE = "Side";
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

double SCALED_INIT_TAU = scaleTime(INIT_TAU);

