from personalFunctions import *


def setAxis(ax, xr, xlabel=r'$t$', ylabel=r'$\lambda_k(t)$'):
    # ax.get_yaxis().set_tick_params(which='both', direction='in', pad = -20)
    # ax.get_xaxis().set_tick_params(which='both', direction='in', pad = -20)

    # ax.set_xlabel(xlabel, fontsize=20)
    # ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim([0, xr])
    ax.set_xticks([0, xr-1])
    ax.set_xticklabels([0, r'$T_p$'], fontsize=13)


if __name__ == '__main__':

    case = 1

    if case == 1:
        """
        plot the local Flouqet exponents of ppo1
        """
        # load data
        expand = np.loadtxt('./ppo/FVexpand1.dat')
        
        fig = plt.figure(figsize=[3, 2])
        ax = fig.add_subplot(111)
        ax.plot(expand[0], lw=1.5, ls='-')
        # ax.plot(expand[1], lw=1.5, ls='-')
        ax.plot(expand[2], lw=1.5, ls='-')
        ax.plot(expand[3], lw=1.5, ls='-')
        ax.plot(expand[4], lw=1.5, ls='-')
        ax.plot(expand[5], lw=1.5, ls='-')
        ax.plot(expand[7], lw=1.5, ls='-')

        ax.plot(expand[8], lw=1.5, ls='-')
        ax.plot(expand[9], lw=1.5, ls='-')

        ax.text(expand.shape[1]/50, 0.4, r'$\lambda_k(t)$', fontsize=18)
        ax.text(expand.shape[1]/3, -0.8, r'$k=1, 3, 4, 5, 6, 8$', fontsize=18)
        ax.text(expand.shape[1]/2, -1.8, r'$k=9, 10$', fontsize=18)
        
        setAxis(ax, expand.shape[1])
        ax.set_ylim([-2.5, 1])
        ax.set_yticks([-2, -1, 0, 1])
        fig.tight_layout(pad=0)
        plt.show()

    if case == 2:
        """
        plot $\lambda_i(t) -\lambda$ for a small subset
        """
        expand = np.loadtxt('./ppo/FVexpand1.dat')
        fe = KSreadFE('../ks22h001t120x64EV.h5', 'ppo', 1)[0]

        fig = plt.figure(figsize=[3, 2])
        ax = fig.add_subplot(111)
        ix = [8, 9, 10, 11]
        for i in ix:
            ax.plot(expand[i]-fe[i], lw=1.5, ls='-', label='k='+str(i+1))

        setAxis(ax, expand.shape[1], ylabel=r'$\lambda_k(t) - \lambda_k$')
        ax.set_yticks([-0.03, 0, 0.03, 0.06])
        ax.legend(loc='best', fontsize=13, frameon=False)
        fig.tight_layout(pad=0)
        plt.show()

        # savez_compressed('ppo1', T=T, nstp=nstp, fe=fe)

    if case == 3:
        """
        plot $\lambda_i(t) -\lambda$ for the remaining set
        """
        expand = np.loadtxt('./ppo/FVexpand1.dat')
        fe = KSreadFE('../ks22h001t120x64EV.h5', 'ppo', 1)[0]

        fig = plt.figure(figsize=[3, 2])
        ax = fig.add_subplot(111)
        ix = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        for i in ix:
            ax.plot(expand[i]-fe[i], lw=1.5, ls='-')

        ax.arrow(expand.shape[1]/20, 0.009, 0, -0.003,
                 width=20, head_length=0.001, head_width=80,
                 fc='k')
        setAxis(ax, expand.shape[1], ylabel=r'$\lambda_k(t) - \lambda_k$')
        ax.set_yticks([-0.008, -0.002, 0.004, 0.01])
        fig.tight_layout(pad=0)
        plt.show()
   
