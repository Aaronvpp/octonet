from config import play_arg, process_arg, global_arg
from dataloader import AcousticDataset, AcousticDataloader
from audio.Audio import AudioPlayer, AudioPlayandRecord
from model.ACF import ACF
from audio.AudioProcessing import KasamiChannelEstimation, chirpChannelEstimation, AudioProcess
import numpy as np
# from matplotlib import pyplot as plt
from time import sleep
from loguru import logger

# >>>>>>>>>>>>>>>>>>>>>>>1. Audio Player / Recorder<<<<<<<<<<<<<<<<<<<<<<<<<
logger.info("Delay {}s".format(global_arg.delay))
sleep(global_arg.delay)
player = None
if global_arg.set_play:
    player = AudioPlayer(play_arg)

elif global_arg.set_playAndRecord:
    player = AudioPlayandRecord(play_arg, path=global_arg.data_path)

if player:
    logger.info("Start playing ...")
    player.begin()
    sleep(play_arg.duration)
    data_record = player.get_record()
    if global_arg.set_save:
        player.save_record()
    # player.end()
    player.end()

# >>>>>>>>>>>>>>>>>>>>>>>2. Data Process<<<<<<<<<<<<<<<<<<<<<<<<<
# if global_arg.set_process:
#     # 2.1 Loading Data recordings
#     data = AcousticDataset(global_arg, file_type="mat")

#     # 2.2 Process Data recordings
#     if process_arg.set_preprocess:
#         preprocess = AudioProcess(data, play_arg)
#         data = preprocess(process_arg=process_arg)

#     # 2.3 Pair Data record / play
#     # todo: modify data_play
#     dataloader = AcousticDataloader(dataset=data, play_arg=play_arg)
#     print(dataloader)

#     # 2.4 Channel Estimation
#     # todo
#     cir = np.array([])
#     cfr = np.array([])
#     for datarec, dataplay, channel_idx in dataloader:
#         if play_arg.wave == "Kasami":
#             channelEstimation = KasamiChannelEstimation(sampling_rate=play_arg.sampling_rate,
#                                                         datarec=datarec,
#                                                         dataplay=dataplay)
#         # print(channelEstimation)
#         # cir_i is CIR for channel i
#         # print(">" * 50 + "Channel " + str(channel_idx) + "<" * 50)

#             cir_i = channelEstimation.CIR
#             logger.debug("cir_i_raw: {}".format(cir_i.shape))
#             #  todo: modify the logic of frames
#             frames = datarec.shape[0] // dataplay.shape[0]
#             if frames > 1:
#                 cir_i = cir_i[-datarec.shape[0] - 1:]
#                 cir_i = cir_i[:dataplay.shape[0] *
#                               frames]
#                 cir_i = cir_i.reshape(frames, dataplay.shape[0])
#                 cir_i = np.transpose(cir_i)
#                 logger.debug("cir_i_reshape: {}".format(cir_i.shape))
#             # logger.debug("cir_i.shape: " + str(cir_i.shape))
#             # plt.plot(cir_i)
#             # plt.show()
#             cir_i = np.expand_dims(cir_i, axis=len(cir_i.shape))
#             cir = cir_i if cir.shape[0] == 0 else np.concatenate(
#                 (cir, cir_i), axis=2)

#             cfr_i = channelEstimation.CFR(cir_i)
#             # normorlize cfr_i
#             cfr_i = cfr_i / np.sum(cfr_i, axis=0)

#             # print("cfr_i.shape: ", cfr_i.shape)
#             cfr = cfr_i if cfr.shape[0] == 0 else np.concatenate(
#                 (cfr, cfr_i), axis=2)

#         elif play_arg.wave == "chirp":
#             # logger.debug("datarec.shape: " + str(datarec.shape))
#             channelEstimation = chirpChannelEstimation(
#                 sampling_rate=play_arg.sampling_rate,
#                 datarec=datarec,
#                 dataplay=dataplay,
#             )
#             cir_i = channelEstimation.CIR.reshape(-1, 1)
#             # logger.debug("cir_i.shape: " + str(cir_i.shape))
#             samples_per_time = play_arg.samples_per_time
#             # logger.debug("samples_per_time: " + str(samples_per_time))
#             # logger.debug("cir.shape: " + str(cir.shape))
#             # frames = (int)(cir_i.shape[0] // samples_per_time)
#             # logger.debug("frames: " + str(frames))
#             # cir_i = cir_i[:frames * samples_per_time
#             #               ].reshape(samples_per_time, frames)
#             cir = cir_i if cir.shape[0] == 0 else np.concatenate(
#                 (cir, cir_i), axis=1)
#             # logger.debug("cir_i.shape: " + str(cir_i.shape))
#             cfr_i = channelEstimation.CFR(cir_i)
#             # logger.debug("cfr_i.shape: " + str(cfr_i.shape))
#             cfr = cfr_i if cfr.shape[0] == 0 else np.concatenate(
#                 (cfr, cfr_i), axis=1)

#     logger.info("cir.shape: " + str(cir.shape))
#     logger.info("cfr.shape: " + str(cfr.shape))

#     # num_channels = cir.shape[2]
#     # acf = ACF(
#     #     CFR=cfr,
#     #     windows_width=args.windows_width,
#     #     windows_step=args.windows_step,
#     #     topK=args.num_topK_subcarriers,
#     #     num_channels=num_channels,
#     #     frames=cir.shape[1])
#     # print(acf)
#     # print(acf.ACF().shape)
# # acf_matrix = acf.ACF()
#     # plt.plot(cir)
#     # plt.show()

#     acf = ACF(
#         CFR=cfr,
#         windows_width=process_arg.windows_width,
#         windows_step=process_arg.windows_step,
#         frames=cfr.shape[1],
#         num_channels=play_arg.nchannels,
#     )
#     motion_statistics = acf.motion_statistics()
#     logger.info("motion.shape: " + str(motion_statistics.shape))
#     # print(motion_statistics)
#     ms_mean = acf.get_motion_curve(motion_statistics)
#     # (time_axis, num_channels)
#     logger.info("ms_mean.shape: " + str(ms_mean.shape))
#     if len(ms_mean.shape) > 1:
#         for channel_idx in range(ms_mean.shape[1]):
#             plt.plot(ms_mean[:, channel_idx])
#     else:
#         plt.plot(ms_mean)
#     plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(plt.figure(figsize=(12, 8)))
# print("acf_matrix[50, 15, :, 0]", acf_matrix[50, 15, :, 0].shape)
# ax.plot_surface(acf_matrix[:, 15, 20, 0].reshape(-1,1), acf_matrix[50, :, 20, 0].reshape(-1,1),
#                 acf_matrix[50, 15, :, 0].reshape(-1,1), rstride=1, cstride=1, cmap='rainbow')
# ax.contourf(acf_matrix[:, 15, 20, 0], acf_matrix[50, :, 20, 0],
#             acf_matrix[50, 15, :, 0], zdir='z', offset=-2, cmap='rainbow')
# plt.show()
