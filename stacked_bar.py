import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np

####Time Decomposition
bottoms = np.zeros((5,))
xs = np.arange(5)

cost_per_stage = np.array([0.22,0.055,0.055,0.055,0.055,0.055,0.055,0.055])/3600  #prices published are per hr


raw_times_per_stage_per_patient = [[[121],[134,115],[116,106,114,108],[119,118,117,119,109,119,114,109],[115,111,106,113,11,114,111,113,105,105,114,108,125,121,122,121]],
                                   [[12],[11,16],[15,14,11,12],[9,10,9,12,14,12,11,10],[10,11,10,13,11,14,13,14,14,10,11,11,16,10,12,10]],
                                   [[17],[21,20],[17,19,18,19],[18,20,22,17,20,18,16,18],[21,16,22,17,16,21,20,21,20,16,16,21,20,16,19,19]],
                                   [[20],[19,23],[21,24,19,16],[22,18,19,19,17,20,22,19],[21,18,18,20,19,24,21,18,21,21,20,19,19,18,18,22]],
                                   [[18],[18,17],[18,17,15,17],[17,18,16,17,18,18,15,16],[16,18,17,18,17,16,19,17,18,16,18,18,19,17,19,18]],
                                   [[481],[444,489],[439,491,410,507],[505,454,489,421,470,407,443,447],[493,444,493,464,458,485,467,443,412,476,460,509,459,450,416,511]],
                                   [[119],[121,123],[115,111,128,121],[112,122,117,120,107,126,118,117],[115,118,122,112,119,132,117,112,120,108,112,122,126,123,119,116]],
                                   [[839],[852,851],[825,821,833,821],[822,829,846,817,832,833,834,821],[819,820,845,841,829,829,832,838,829,833,830,824,847,814,817,819]]]

average_times_per_stage = [[np.mean(raw_times_per_stage_per_patient[i][j])for j in range(5)] for i in range(len(raw_times_per_stage_per_patient))]
average_cost = [np.array(average_times_per_stage[i])*cost_per_stage[i] for i in range(len(average_times_per_stage))]
average_cost = np.stack(average_cost,axis=0)
complete_cost = average_cost.sum(axis=0)

# for i in range(len(raw_times_per_stage_per_patient)):
#     plt.bar(x=xs,width = 0.75,bottom=bottoms,height=average_times_per_stage[i],label=f"Stage {i+1}")
#     bottoms += average_times_per_stage[i]
# plt.legend()
# # plt.xlabel("Participants")
# plt.ylabel("Average Exectution Time per Participant [s]")
# plt.xticks(ticks=xs, labels=['1 Participant', '2 Participants', '4 Participants', '8 Participants', '16 Participants'],rotation=90)
# # plt.ylim(0,np.max(bottoms)*1.25)
# plt.xlim(-1,6)
# plt.tight_layout()
# plt.savefig("time_decomposition.png",dpi=450)
# plt.show()

####Weak Scaling Their Stuff
# walltimes_weak = np.array([272.104,336.691,647.817,1110.247,1905.763])
sum_times = np.sum(np.stack(average_times_per_stage,axis=0),axis=0)

fig,axs = plt.subplots(1,1,figsize=(6,5))
for i in range(len(average_times_per_stage)):
    axs.plot(np.arange(average_times_per_stage[i].__len__()),average_times_per_stage[i],'o-',label=f"Stage {i+1}") 
axs.plot(np.arange(average_times_per_stage[i].__len__()),sum_times,'o-',color=(0,0,0),label=f"Total") 
axs.set_xticks(np.arange(average_times_per_stage[i].__len__()), labels=[1,2,4,8,16])
axs.set_xlabel("\# Participants")
axs.set_ylabel("Average Runtime")
axs.set_title("Weak Scaling")
axs.legend()

fig.tight_layout()

fig.savefig("weak_scalability.png",dpi=450)
plt.show()
print("stop")


# ####Scaling Our Stuff
# walltimes_weak = np.array([272.104,336.691,647.817,1110.247,1905.763]) / 4
# walltimes_strong = np.array([531.849,336.691,230.549,214.958,217.646])
# speedup = walltimes_strong[0]/walltimes_strong
# efficiency = (speedup / np.array([1,2,4,8,16]))[1:]

# fig,axs = plt.subplots(2,2,figsize=(8,6))
# axs = axs.flatten()

# axs[0].plot(np.arange(walltimes_strong.shape[0]),walltimes_strong,'o-',color=(0,0,0))
# axs[0].set_xticks(np.arange(walltimes_strong.shape[0]), labels=[1,2,4,8,16])
# axs[0].set_xlabel("\#cores")
# axs[0].set_ylabel("Time for 8 FASTQ Files [s]")
# axs[0].set_title("Strong Scaling")

# axs[1].plot(np.arange(walltimes_weak.shape[0]),walltimes_weak,'o-',color=(0,0,0))
# axs[1].set_xticks(np.arange(walltimes_weak.shape[0]), labels=[1,2,4,8,16])
# axs[1].set_xlabel("\#cores and \#FASTQ files")
# axs[1].set_ylabel("Time per FASTQ File per Core [s]")
# axs[1].set_title("Weak Scaling")

# axs[2].plot(np.arange(speedup.shape[0]),speedup,'o-',color=(0,0,0))
# axs[2].set_xticks(np.arange(speedup.shape[0]), labels=[1,2,4,8,16])
# axs[2].set_xlabel("\#cores")
# axs[2].set_ylabel("Speedup")
# axs[2].set_title("Speedup")

# axs[3].plot(np.arange(efficiency.shape[0])+1,efficiency,'o-',color=(0,0,0))
# axs[3].set_xticks(np.arange(efficiency.shape[0]+1), labels=[None,2,4,8,16])
# axs[3].set_xlabel("\#cores")
# axs[3].set_ylabel("Efficiency")
# axs[3].set_title("Efficiency")

# fig.tight_layout()

# fig.savefig("microbiome_pipeline_scalability.png",dpi=450)
# plt.show()
print("sho")