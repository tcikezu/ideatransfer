""" simulate returns an animated 3d scatter plot of the community allIdeas object.
The input parameters are the community, the interaction coefficient gamma,
and also fn, a string file-name which defaults to date and time."""
def simulate(X=None, gamma = 0.005, T = 80,fn=date_time):
    # seems like the same community is used again and again if I do not specify None case below.
    if X is None:
        X = community(300,3)
    fps = 40

    # Data to store X.allIdeas, to then make animation
    dataX = np.zeros((T*fps,X.numberMembers))
    dataY = np.zeros((T*fps,X.numberMembers))
    dataZ = np.zeros((T*fps,X.numberMembers))

    # Iterate the idea transfer throughout community
    for t in range(0,T+1):
        np.random.seed()
        dataX[t,:] = X.allIdeas[:,0]
        dataY[t,:] = X.allIdeas[:,1]
        dataZ[t,:] = X.allIdeas[:,2]
        #transfer.deterministicMerge(X, gamma/(t+1)**0.5)
        transfer.probabilisticMerge(X, gamma,t)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(dataX[0], dataY[0], dataZ[0])

    axTicks = np.linspace(-c.domainSize,c.domainSize,4)

    #animation function for animation.FuncAnimation
    def update(ifrm,dataX,dataY,dataZ):
        ax.clear()
        plt.autoscale(False)
        ax.set_xticks(axTicks)
        ax.set_yticks(axTicks)
        ax.set_zticks(axTicks)
        ax.scatter3D(dataX[ifrm], dataY[ifrm], dataZ[ifrm])
        ax.set_xlabel("frame: %d" % (ifrm))

    ani = animation.FuncAnimation(fig, update, T, fargs=(dataX,dataY,dataZ),interval = T/fps )
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)

    #plt.show()

    return (dataX,dataY,dataZ)
