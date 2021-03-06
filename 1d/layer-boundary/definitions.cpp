#include "definitions.h"

double CustomExactFunction::uhat(double x) 
{
  return 1. - (exp(K*x) + exp(-K*x)) / (exp(K) + exp(-K));
}

double CustomExactFunction::duhat_dx(double x) 
{
  return -K * (exp(K*x) - exp(-K*x)) / (exp(K) + exp(-K));
}

double CustomExactFunction::dduhat_dxx(double x) 
{
  return -K*K * (exp(K*x) + exp(-K*x)) / (exp(K) + exp(-K));
}

CustomExactSolution::CustomExactSolution(Mesh* mesh, double K) : ExactSolutionScalar<double>(mesh)
{
  cef = new CustomExactFunction(K);
}

CustomExactSolution::~CustomExactSolution() 
{ 
  delete cef;
}

double CustomExactSolution::value (double x, double y) const 
{
  return cef->uhat(x) * cef->uhat(y);
}

void CustomExactSolution::derivatives(double x, double y, double& dx, double& dy) const 
{
  dx = cef->duhat_dx(x) * cef->uhat(y);
  dy = 0;
}

Ord CustomExactSolution::ord(Ord x, Ord y) const 
{
  return Ord(20);
}


CustomFunction::CustomFunction(double coeff1) : Hermes::Hermes2DFunction<double>(), coeff1(coeff1) 
{
  cef = new CustomExactFunction(coeff1);
}

CustomFunction::~CustomFunction() 
{ 
  delete cef;
}

double CustomFunction::value(double x, double y) const 
{
  return -(-cef->dduhat_dxx(x) + coeff1 * coeff1 * cef->uhat(x));
}

Ord CustomFunction::value_ord(Ord x, Ord y) const 
{
  return Ord(5);
}


CustomWeakForm::CustomWeakForm(CustomFunction* f) : WeakForm<double>(1) 
{
  // Jacobian.
  add_matrix_form(new Hermes::Hermes2D::WeakFormsH1::DefaultJacobianDiffusion<double>(0, 0));
  add_matrix_form(new Hermes::Hermes2D::WeakFormsH1::DefaultMatrixFormVol<double>(0, 0, Hermes::HERMES_ANY, 
                                                        new Hermes::Hermes2DFunction<double>(f->coeff1*f->coeff1)));
  // Residual.
  add_vector_form(new Hermes::Hermes2D::WeakFormsH1::DefaultResidualDiffusion<double>(0));
  add_vector_form(new Hermes::Hermes2D::WeakFormsH1::DefaultResidualVol<double>(0, Hermes::HERMES_ANY, 
                                                      new Hermes::Hermes2DFunction<double>(f->coeff1*f->coeff1)));
  add_vector_form(new Hermes::Hermes2D::WeakFormsH1::DefaultVectorFormVol<double>(0, Hermes::HERMES_ANY, f));
}
