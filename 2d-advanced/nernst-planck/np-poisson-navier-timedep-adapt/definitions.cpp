#include "definitions.h"

class ScaledWeakFormPNPEulerCranic : public WeakForm<double> {
public:
  ScaledWeakFormPNPEulerCranic(double* tau, double epsilon, double AC0, double mu, double lambda, double l,
        Solution<double>* C_prev_time, Solution<double>* phi_prev_time) : WeakForm<double>(4) {
      for(unsigned int i = 0; i < 4; i++) {
        ScaledWeakFormPNPEulerCranic::Residual* vector_form =
            new ScaledWeakFormPNPEulerCranic::Residual(i, tau, epsilon, AC0, mu, lambda, l);
        if(i == 0) {
          vector_form->ext.push_back(C_prev_time);
          vector_form->ext.push_back(phi_prev_time);
        }
        add_vector_form(vector_form);
        for(unsigned int j = 0; j < 4; j++)
          add_matrix_form(new ScaledWeakFormPNPEulerCranic::Jacobian(i, j, tau, epsilon, AC0, mu, lambda, l));
      }
    };

private:
  class Jacobian : public MatrixFormVol<double> {
  public:
    Jacobian(int i, int j, double* tau, double epsilon, double AC0, 
        double mu, double lambda, double l) : MatrixFormVol<double>(i, j),
          i(i), j(j), tau(tau), epsilon(epsilon), AC0(AC0), mu(mu), lambda(lambda), l(l) {}

    template<typename Real, typename Scalar>
    Real matrix_form(int n, double *wt, Func<Scalar> *u_ext[], Func<Real> *u,
                       Func<Real> *v, Geom<Real> *e, ExtData<Scalar> *ext) const {
      Real result = Real(0);
      Func<Scalar>* prev_newton;
      switch(i * 10 + j) {
        case 0:
          prev_newton = u_ext[1];
          for (int i = 0; i < n; i++) {

            result += wt[i] * (u->val[i] * v->val[i] / *(this->tau) +
                this->epsilon * 0.5 * ((u->dx[i] * v->dx[i] + u->dy[i] * v->dy[i]) +
                    u->val[i] * (prev_newton->dx[i] * v->dx[i] + prev_newton->dy[i] * v->dy[i])));
          }
          return result;
          break;
        case 1:
          prev_newton = u_ext[0];
          for (int i = 0; i < n; i++) {
            result += wt[i] * (0.5 * this->epsilon * prev_newton->val[i] * (u->dx[i] * v->dx[i] + u->dy[i] * v->dy[i]));
          }
          return result;
          break;
        case 10:
          for (int i = 0; i < n; i++) {
            result += wt[i] * ( -1.0/(2 * this->epsilon * this->epsilon) * u->val[i] * v->val[i]);
          }
          return result;
          break;
        case 11: for (int i = 0; i < n; i++) {
            result += wt[i] * ( u->dx[i] * v->dx[i] + u->dy[i] * v->dy[i]);
          }
          return result;
          break;
        case 20:
          for (int i = 0; i < n; i++) { // XXX check -
            result += wt[i] * (-this->l / this->mu * this->AC0 * u->val[i] * v->val[i]);
          }
          return result;
          break;
        case 22:
          for (int i = 0; i < n; i++) {
            result += wt[i] * ((this->lambda / this->mu + 2) * u->dx[i] * v->dx[i] +
              u->dy[i] * v->dy[i]);
          }
          return result;
          break;
        case 23:
          for (int i = 0; i < n; i++) {
            result += wt[i] * (u->dx[i] * v->dy[i] + (this->lambda / this->mu) * u->dy[i] * v->dx[i]);
          }
          return result;
          break;
        case 32:
          for (int i = 0; i < n; i++) {
            result += wt[i] * (u->dy[i] * v->dx[i] + (this->lambda / this->mu) * u->dx[i] * v->dy[i]);
          }
          return result;
          break;
        case 33:
          for (int i = 0; i < n; i++) {
            result += wt[i] * ((this->lambda / this->mu + 2) * u->dy[i] * v->dy[i] +
              u->dx[i] * v->dx[i]);
          }
          return result;
          break;
        default:
          return result;
      }
    }

    virtual double value(int n, double *wt, Func<double> *u_ext[], Func<double> *u,
                 Func<double> *v, Geom<double> *e, ExtData<double> *ext) const {
      return matrix_form<double, double>(n, wt, u_ext, u, v, e, ext);
    }

    virtual Ord ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *u, Func<Ord> *v,
            Geom<Ord> *e, ExtData<Ord> *ext) const {
      return matrix_form<Ord, Ord>(n, wt, u_ext, u, v, e, ext);
    }

    // Members.
    int i, j;
    double* tau;
    double epsilon;
    double lambda;
    double mu;
    double AC0;
    double l;
  };

  class Residual : public VectorFormVol<double>
      {
      public:
        Residual(int i, double* tau, double epsilon, double AC0, double mu, double lambda, double l)
          : VectorFormVol<double>(i), i(i), tau(tau), epsilon(epsilon), AC0(AC0), mu(mu), lambda(lambda), l(l) {}

        template<typename Real, typename Scalar>
        Real vector_form(int n, double *wt, Func<Scalar> *u_ext[],
                            Func<Real> *v, Geom<Real> *e, ExtData<Scalar> *ext) const {
          Real result = Real(0);
          Func<Scalar>* C_prev_time;
          Func<Scalar>* phi_prev_time;
          Func<Scalar>* C_prev_newton;
          Func<Scalar>* phi_prev_newton;
          Func<Scalar>* U1_prev_newton;
          Func<Scalar>* U2_prev_newton;
          switch(i) {
            case 0:
              C_prev_time = ext->fn[0];
              phi_prev_time = ext->fn[1];
              C_prev_newton = u_ext[0];
              phi_prev_newton = u_ext[1];
              for (int i = 0; i < n; i++) {
                result += wt[i] * ((C_prev_newton->val[i] - C_prev_time->val[i]) * v->val[i] / *(this->tau) +
                    0.5 * this->epsilon * ((C_prev_newton->dx[i] * v->dx[i] + C_prev_newton->dy[i] * v->dy[i]) +
                    (C_prev_time->dx[i] * v->dx[i] + C_prev_time->dy[i] * v->dy[i]) +
                      C_prev_newton->val[i] * (phi_prev_newton->dx[i] * v->dx[i] + phi_prev_newton->dy[i] * v->dy[i]) +
                      C_prev_time->val[i] * (phi_prev_time->dx[i] * v->dx[i] + phi_prev_time->dy[i] * v->dy[i])));
              }
              return result;
            case 1:
              C_prev_newton = u_ext[0];
              phi_prev_newton = u_ext[1];
              for (int i = 0; i < n; i++) {
                result += wt[i] * ((phi_prev_newton->dx[i] * v->dx[i] + phi_prev_newton->dy[i] * v->dy[i]) +
                    v->val[i] * 1 / (2 * this->epsilon * this->epsilon) * (1 - C_prev_newton->val[i]));
              }
              return result;
            case 2:
              C_prev_newton = u_ext[0];
              phi_prev_newton = u_ext[1];
              U1_prev_newton = u_ext[2];
              U2_prev_newton = u_ext[3];
              for (int i = 0; i < n; i++) {
                result += wt[i] * ((this->lambda / this->mu + 2) * (U1_prev_newton->dx[i] * v->dx[i]) 
                    + U1_prev_newton->dy[i] * v->dy[i] + U2_prev_newton->dx[i]*v->dy[i] 
                    + (this->lambda / this->mu) * (U2_prev_newton->dy[i] * v->dx[i])
                    + (this->l / this->lambda * this->AC0) * (1 - C_prev_newton->val[i]) * v->val[i]);
              }
              return result;
            case 3:
              C_prev_newton = u_ext[0];
              phi_prev_newton = u_ext[1];
              U1_prev_newton = u_ext[2];
              U2_prev_newton = u_ext[3];
              for (int i = 0; i < n; i++) {
                result += wt[i] * ((this->lambda / this->mu + 2)* (U2_prev_newton->dy[i] * v->dy[i]) 
                    + U2_prev_newton->dx[i] * v->dx[i] + U1_prev_newton->dy[i]*v->dx[i] 
                    + (this->lambda / this->mu) * (U1_prev_newton->dx[i] * v->dy[i]));
              }
              return result;
            default:
              return result;
          }
        }

        virtual double value(int n, double *wt, Func<double> *u_ext[],
                     Func<double> *v, Geom<double> *e, ExtData<double> *ext) const {
          return vector_form<double, double>(n, wt, u_ext, v, e, ext);
        }

        virtual Ord ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *v,
                Geom<Ord> *e, ExtData<Ord> *ext) const {
          return vector_form<Ord, Ord>(n, wt, u_ext, v, e, ext);
        }

        // Members.
        int i;
        double* tau;
        double epsilon;
        double AC0;
        double mu;
        double lambda;
        double l;
      };



};

