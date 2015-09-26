from personalFunctions import *


def setAxis(ax, xr, xlabel=r'$t$', ylabel=r'$\lambda_k(t)$'):
    # ax.get_yaxis().set_tick_params(which='both', direction='in', pad = -20)
    # ax.get_xaxis().set_tick_params(which='both', direction='in', pad = -20)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim([0, xr])
    ax.set_xticks([0, xr-1])
    ax.set_xticklabels([0, r'$T_p$'])


if __name__ == '__main__':

    case = 1

    if case == 1:
        """
        plot the local Flouqet exponents of ppo1
        """
        # load data
        a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', 'ppo', 1)
        expand = np.log(np.loadtxt('./ppo/FVexpand1.dat')) / (5 * T / nstp)
        
        fig = plt.figure(figsize=[6, 5])
        ax = fig.add_subplot(111)
        ax.plot(expand[0], lw=1.5, ls='-')
        ax.plot(expand[2], lw=1.5, ls='-')
        ax.plot(expand[3], lw=1.5, ls='-')
        ax.plot(expand[4], lw=1.5, ls='-')
        ax.plot(expand[5], lw=1.5, ls='-')
        ax.plot(expand[7], lw=1.5, ls='-')

        ax.plot(expand[8], lw=1.5, ls='-')
        ax.plot(expand[9], lw=1.5, ls='-')

        ax.text(expand.shape[1]/3, -0.6, r'$k=1, 3, 4, 5, 6, 8$', fontsize=20)
        ax.text(expand.shape[1]/2, -1.8, r'$k=9, 10$', fontsize=20)
        
        setAxis(ax, expand.shape[1])
        fig.tight_layout(pad=0)
        plt.show()

    if case == 2:
        """
        plot $\lambda_i(t) -\lambda$ for a small subset
        """
        a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', 'ppo', 1)
        expand = np.log(np.loadtxt('./ppo/FVexpand1.dat')) / (5 * T / nstp)
        fe = KSreadFE('../ks22h001t120x64EV.h5', 'ppo', 1)[0]

        fig = plt.figure(figsize=[6, 5])
        ax = fig.add_subplot(111)
        ix = [8, 9, 10, 11]
        for i in ix:
            ax.plot(expand[i]-fe[i], lw=1.5, ls='-', label='k='+str(i+1))

        setAxis(ax, expand.shape[1], ylabel=r'$\lambda_k(t) - \lambda_k$')
        ax.legend(loc='best')
        fig.tight_layout(pad=0)
        plt.show()

    if case == 3:
        """
        plot $\lambda_i(t) -\lambda$ for the remaining set
        """
        a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', 'ppo', 1)
        expand = np.log(np.loadtxt('./ppo/FVexpand1.dat')) / (5 * T / nstp)
        fe = KSreadFE('../ks22h001t120x64EV.h5', 'ppo', 1)[0]

        fig = plt.figure(figsize=[6, 5])
        ax = fig.add_subplot(111)
        ix = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        for i in ix:
            ax.plot(expand[i]-fe[i], lw=1.5, ls='-')

        ax.arrow(expand.shape[1]/20, 0.009, 0, -0.003,
                 width=2, head_length=0.001, head_width=40,
                 fc='k')
        setAxis(ax, expand.shape[1], ylabel=r'$\lambda_k(t) - \lambda_k$')
        fig.tight_layout(pad=0)
        plt.show()

    if case == 4:
        """
        The partial hyperbolicity degree
        """
        pn = np.zeros(29)
        for i in range(1, 201):
            print i
            expand = np.loadtxt('./ppo/FVexpand' + str(i) + '.dat')
            for j in range(1, 30):
                x = np.amin(expand[:j], axis=0) - np.amax(expand[j:], axis=0)
                if np.amin(x) > 0:
                    pn[j-1] += 1

        rn = np.zeros(29)
        for i in range(1, 201):
            print i
            expand = np.loadtxt('./rpo/FVexpand' + str(i) + '.dat')
            for j in range(1, 30):
                x = np.amin(expand[:j], axis=0) - np.amax(expand[j:], axis=0)
                if np.amin(x) > 0:
                    rn[j-1] += 1

        ph = pn + rn

        fig = plt.figure(figsize=[6, 5])
        ax = fig.add_subplot(111)
        ax.scatter(range(1, 30), ph, s=30, facecolor='b', edgecolors='none')
        ax.set_xticks(range(0, 31, 2))
        ax.set_xlim([0, 30])
        # ax.set_ylim([-200, 600])
        ax.set_xlabel('k', fontsize=20)
        ax.grid('on')
        fig.tight_layout(pad=0)
        plt.show()

    
