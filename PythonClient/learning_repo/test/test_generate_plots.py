import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#generatedata
def getdata( ):
    basecond=[[18,20,19,18,13,4,1],
            [20,17,12,9,3,0,0],
            [20,20,20,12,5,3,0]]

    cond1=[[18,19,18,19,20,15,14],
    [19,20,18,16,20,15,9],
    [19,20,20,20,17,10,0],
    [20,20,20,20,7,9,1]]

    cond2=[[20,20,20,20,19,17,4],
    [20,20,20,20,20,19,7],
    [19,20,20,19,19,15,2]]

    cond3=[[20,20,20,20,19,17,12],
    [18,20,19,18,13,4,1],
    [20,19,18,17,13,2,0],
    [19,18,20,20,15,6,0]]

    return basecond,cond1,cond2,cond3

# load data
results=getdata()
fig=plt.figure()

# We will plot iterations 0...6
xdata=np.array([0,1,2,3,4,5,6])/5.

#Plot each line
#(maywanttoautomatethisparte.g.withaloop).

sns.tsplot(time=xdata,data=results[0],color='r',linestyle='-',condition='a')
sns.tsplot(time=xdata,data=results[1],color='g',linestyle='--')
# sns.tsplot(time=xdata,data=results[2],color='b',linestyle=':')
# sns.tsplot(time=xdata,data=results[3][0],color='k',linestyle='-.')
# ax.set_label = 'a'

#Oury−axisis"successrate"here.
plt.ylabel("SuccessRate",fontsize='medium')
#Ourx−axisisiterationnumber.
plt.xlabel("IterationNumber",fontsize='medium')
#Ourtaskiscalled"AwesomeRobotPerformance"
plt.title("AwesomeRobotPerformance",fontsize='medium')
#Legend
plt.legend(loc='bottomleft')
# #Showtheplotonthescreen.
plt.show()


# ##### REGULAR WAY NOW
# # load style sheet
#
# plt.style.use("dwplot")
# print(results[0][0])
#
# plt.figure(2)
# plt.plot(xdata,results[0][0],'-r')
# plt.plot(xdata,results[1][0],'--g')
# plt.plot(xdata,results[2][0],':b')
# plt.plot(xdata,results[3][0],'-.k')
#
# #Oury−axisis"successrate"here.
# plt.ylabel("SuccessRate",fontsize='medium')
# #Ourx−axisisiterationnumber.
# plt.xlabel("IterationNumber",fontsize='medium')
# #Ourtaskiscalled"AwesomeRobotPerformance"
# plt.title("AwesomeRobotPerformance",fontsize='medium')
# #Legend.
# plt.legend(loc='bottomleft')
# #Showtheplotonthescreen.
# plt.show()
