# Number of one-dimensional standard normal random variables used by the ML-based approximation method
$$
    \begin{split}
        \frac{1}{J_m}\sum_{j=1}^{J_m}
        \bbbbbr{
        \bV^{j,\mathbf{s}}_n\bpr{\theta,\Y^{n,m,j}_{N-n}(\omega)}
        -
        \bV^{j,\mathbf{s}}_{n-1}\bpr{\Theta^{n-1}_{M_{n-1}}(\omega),\Y^{n,m,j}_{N-n+1}(\omega)}\\
        & - \tfrac{(t_n-t_{n-1})}{K_n} \bbbbr{ \textstyle \sum \limits_{k=1}^{K_n}  f\bbpr{t_{n-1},
        %& \quad \quad \cdot
        \Y^{n,m,j}_{N-n+1}(\omega),
        Z_{ \mathcal{Y}^{n,m,j}_{ N - n + 1 }(\omega), k }^{ n, m,j }(\omega),\\
        & \bV^{j,\mathbf{s}}_{n-1}\bpr{\Theta^{n-1}_{M_{n-1}}(\omega),\Y^{n,m,j}_{N-n+1}(\omega)},
        \bV^{j,\mathbf{s}}_{n-1}\bpr{\Theta^{n-1}_{M_{n-1}}(\omega),	Z_{ \mathcal{Y}^{n,m,j}_{ N - n + 1 }(\omega), k }^{ n, m,j }(\omega)}
        %,\\
        %& (\nabla_x \bV^{j,\mathbf{s}}_{n-1})\bpr{\Theta^{n-1}_{M_{n-1}}(\omega),\Y^{n,m,j}_{N-n+1}(\omega)},
        %(\nabla_x \bV^{j,\mathbf{s}}_{n-1})\bpr{\Theta^{n-1}_{M_{n-1}}(\omega),Z_{ \mathcal{Y}^{n,m,j}_{ N - n + 1 }, k }^{ n, m,j }(\omega)}
        }}
        }^2,
        % \\
        %& = \Big| \bV^{j,\mathbf{s}}_n\bpr{\theta,\Y^{n,m}_{N-n}(\omega)} - \mathfrak{V}^{n,\Theta^{n-1}_{M_{n-1}}(\omega)}\bpr{\Y^{n,m}_{N-n+1}(\omega)} \Big|^2,
        %\end{split}
    \end{split}
$$

# Number of one-dimensional standard normal random variables used by the MLP
$$
	\begin{split}
		&
		U^\dindex_{n,M,r}(t,x) 
		= 
		\Biggl[\sum_{l=0}^{n-1} \frac{(T-t)}{M^{n-l}}  
		\sum_{m=1}^{M^{n-l}} \frac{1}{K_{n,l,m}}
		\sum_{k=1}^{K_{n,l,m}}
		\bbbbr{ f \bbpr{
				V_t^{(\dindex,l,m)},
				X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}},
				\Zz^{(\dindex,l,m,k) }_{ X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}} },\\
			&\quad 
				\phi_{r}\bbpr{U^{(\dindex,l,m)}_{l,M,r}\bpr{V_t^{(\dindex,l,m)},X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}}}},
				\phi_{r}\bbpr{U^{(\dindex,l,m)}_{l,M,r}\bpr{V_t^{(\dindex,l,m)},\Zz^{(\dindex,l,m,k) }_{ X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}}}}}
			} \\
			&\quad- \mathbbm{1}_\N(l) \, f \bbpr{
				V_t^{(\dindex,l,m)},
				X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}}, 
				\Zz^{(\dindex,l,m,k)}_{ X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}} },
				\phi_{r}\bbpr{U^{(\dindex,l,-m)}_{\max\{l-1,0\},M,r}\bpr{V_t^{(\dindex,l,m)},X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}}}},\\
			&\quad 
				\phi_{r}\bbpr{U^{(\dindex,l,-m)}_{\max\{l-1,0\},M,r}\bpr{V_t^{(\dindex,l,m)},\Zz^{(\dindex,l,m,k) }_{ X^{x,(\dindex,l,m)}_{t,V_t^{(\dindex,l,m)}} }}}
			}}\Biggr] 
			+  
			\frac{\1_{\N}(n)}{M^n} \bbbbbr{\sum_{m=1}^{M^n} g\bpr{X^{x,(\dindex,0,-m)}_{t,T}}  },
	\end{split}
$$