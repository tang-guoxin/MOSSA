## MOSSA

### I.Introduction



### II. Basic Concepts

- $Def1$ **(全局最小值)** 给定一个函数$f:\Omega  \subseteq {R^n} \to R$，$\Omega  \ne \emptyset $，我们称$f\left( {{{\mathop x\limits^ \to  }^*}} \right) >  - \infty $是$f(x)$的一个全局最小值，如果

$$
\forall \mathop x\limits^ \to   \in \Omega :f\left( {{{\mathop x\limits^ \to  }^*}} \right) \le f\left( {\mathop x\limits^ \to  } \right)
$$

其中，${{{\mathop x\limits^ \to  }^*}}$是一个全局最优解，$f$是目标函数，$\Omega$是${{{\mathop x\limits^ \to  }}}$的可行域($\Omega\subseteq S$)，$S$为搜索空间。

- $Def2$  **(一般的多目标问题[MOP])** 设向量${{{\mathop x\limits^ \to  }^*}}=[x_1^*,x_2^*,...,x_n^*]$是向量函数

$$
\mathop f\limits^ \to  \left( {\mathop x\limits^ \to  } \right) = \left[ {{f_1}\left( {\mathop x\limits^ \to  } \right),{f_2}\left( {\mathop x\limits^ \to  } \right),...,{f_k}\left( {\mathop x\limits^ \to  } \right)} \right]
$$

的一个全局最小值解，它满足$m$个不等式约束${g_i}\left( {\mathop x\limits^ \to  } \right) \ge 0,i = 1,2,...,m$和$p$个等式约束${h_i}\left( {\mathop x\limits^ \to  } \right) = 0,i = 1,2,...,p$。其中${\mathop x\limits^ \to  }=[x_1,x_2,...,x_n]$是决策变量向量。

- $Def3$  **(Pareto最优)**  我们称${{{\mathop x\limits^ \to  }^*}}\in \Omega$是Pareto最优，如果对于每一个${{{\mathop x\limits^ \to  }}}\in \Omega$，$I=\{{1,2,...,k}\}$，必然满足$(3)$式中的其中一个条件。

$$
\left\{ \begin{array}{l}
{\forall _{i \in I}}\left( {{f_i}\left( {\mathop x\limits^ \to  } \right) = {f_i}\left( {{{\mathop x\limits^ \to  }^*}} \right)} \right)\\
{\exists _{i \in I}}\left( {{f_i}\left( {\mathop x\limits^ \to  } \right) > {f_i}\left( {{{\mathop x\limits^ \to  }^*}} \right)} \right)
\end{array} \right.
$$

换句话说，Pareto最优





$$
{{{\mathop x\limits^ \to  }^*}}
$$
