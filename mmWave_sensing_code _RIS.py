import serial
import time
import numpy as np
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsRectItem  # Correct import for bounding box item.
import pyqtgraph as pg
from sklearn.cluster import DBSCAN
from parser_mmw_demo import parser_one_mmw_demo_output_packet

DEBUG = False

# Global constants and variables.
maxBufferSize = 2**15
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
currentIndex = 0
word = [1, 2**8, 2**16, 2**24]

# Inline sensor configuration commands.
configCommands = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 60 359 7 57.14 0 0 70 1 256 5209 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 4",
    "chirpCfg 2 2 0 0 0 0 0 2",
    "frameCfg 0 2 16 0 100 1 0",
    "lowPower 0 0",
    "guiMonitor -1 1 1 0 0 0 1",
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 4 2 3 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "bpmCfg -1 0 0 1",
    "lvdsStreamCfg -1 0 0 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 5 121 0",
    "CQSigImgMonitor 0 127 4",
    "analogMonitor 0 0",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0 8.92",
    "cfarFovCfg -1 1 -1 1.00",
    "calibData 0 0 0",
    "sensorStart"
]

# RIS mapping dictionaries.
positive_image_data = {
    7: 3, 8: 6, 9: 9, 10: 12, 11: 15, 12: 18, 13: 21, 14: 24, 15: 27, 16: 30,
    17: 33, 18: 36, 19: 39, 20: 42, 21: 45, 22: 48, 23: 51, 24: 54, 25: 57, 26: 60
}
negative_image_data = {
    38: -3, 39: -6, 40: -9, 41: -12, 42: -15, 43: -18, 44: -21, 45: -24,
    46: -27, 47: -30, 48: -33, 49: -36, 50: -39, 51: -42, 52: -45, 53: -48,
    54: -51, 55: -54, 56: -57, 57: -60
}

def find_closest_image(angle_hor):
    image_data = positive_image_data if angle_hor >= 0 else negative_image_data
    closest_image = None
    min_diff = float('inf')
    for image, theta in image_data.items():
        diff = abs(theta - angle_hor)
        if diff < min_diff:
            min_diff = diff
            closest_image = image
    return closest_image

def send_image_to_serial(image_index):
    port = serial.Serial(port='COM5', baudrate=9600, timeout=10)
    port.set_buffer_size(rx_size=1, tx_size=1)
    port.reset_input_buffer()
    port.reset_output_buffer()
    if not port.is_open:
        port.open()
    port.write(bytearray([image_index]))
    time.sleep(10e-3)
    port.close()

# Helper function to flip horizontal axis (multiply x values by -1).
def flip_x(points):
    new_points = points.copy()
    new_points[:, 0] = -new_points[:, 0]
    return new_points

###################
# Kalman Filter Class 
####################
class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((5, 1))
        self.F = np.array([[1, 0, dt,  0,  0],
                           [0, 1,  0, dt,  0],
                           [0, 0,  1,  0,  0],
                           [0, 0,  0,  1,  0],
                           [0, 0,  0,  0,  1]])
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1]])
        self.P = np.eye(5) * 500
        self.Q = np.eye(5) * 1.0
        self.R = np.eye(3) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.reshape(z, (3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

####################
# Data parsing functions 
####################
def serialConfig():
    global CLIport, Dataport
    CLIport = serial.Serial('COM6', 115200)
    Dataport = serial.Serial('COM7', 921600)
    for command in configCommands:
        CLIport.write((command + '\n').encode())
        time.sleep(0.01)
    return CLIport, Dataport

def parseConfigFile(_):
    configParameters = {}
    for line in configCommands:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "profileCfg":
            configParameters['startFreq'] = float(tokens[2])
            configParameters['idleTime'] = float(tokens[3])
            configParameters['rampEndTime'] = float(tokens[5])
            configParameters['freqSlopeConst'] = float(tokens[8])
            configParameters['numAdcSamples'] = int(tokens[10])
            configParameters['digOutSampleRate'] = int(tokens[11])
        elif tokens[0] == "frameCfg":
            configParameters['chirpStartIdx'] = int(tokens[1])
            configParameters['chirpEndIdx'] = int(tokens[2])
            configParameters['numLoops'] = int(tokens[3])
            configParameters['numFrames'] = int(tokens[4])
            configParameters['framePeriodicity'] = float(tokens[5])
    numAdcSamples = configParameters.get('numAdcSamples', 0)
    numAdcSamplesRoundTo2 = 1
    while numAdcSamples > numAdcSamplesRoundTo2:
        numAdcSamplesRoundTo2 *= 2
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    numChirpsPerFrame = (configParameters['chirpEndIdx'] - configParameters['chirpStartIdx'] + 1) * configParameters['numLoops']
    configParameters["numDopplerBins"] = numChirpsPerFrame / 3
    configParameters["rangeResolutionMeters"] = (3e8 * configParameters['digOutSampleRate'] * 1e3) / (
        2 * configParameters['freqSlopeConst'] * 1e12 * configParameters['numAdcSamples'])
    configParameters["rangeIdxToMeters"] = (3e8 * configParameters['digOutSampleRate'] * 1e3) / (
        2 * configParameters['freqSlopeConst'] * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * configParameters['startFreq'] * 1e9 *
                                                        (configParameters['idleTime'] + configParameters['rampEndTime']) *
                                                        1e-6 * configParameters["numDopplerBins"] * 3)
    configParameters["maxRange"] = (300 * 0.9 * configParameters['digOutSampleRate']) / (
        2 * configParameters['freqSlopeConst'] * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * configParameters['startFreq'] * 1e9 *
                                             (configParameters['idleTime'] + configParameters['rampEndTime']) *
                                             1e-6 * 3)
    return configParameters

def readAndParseData14xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    magicOK = 0
    dataOK = 0
    frameNumber = 0
    detObj = {}
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength+byteCount] = byteVec[:byteCount]
        byteBufferLength += byteCount
    if byteBufferLength > 16:
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
        if startIdx:
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype='uint8')
                byteBufferLength -= startIdx[0]
            if byteBufferLength < 0:
                byteBufferLength = 0
            totalPacketLen = np.matmul(byteBuffer[12:12+4], word)
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    if magicOK:
        readNumBytes = byteBufferLength
        allBinData = byteBuffer
        totalBytesParsed = 0
        numFramesParsed = 0
        parser_result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, \
            detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, \
            detectedAzimuth_array, detectedElevation_array, detectedSNR_array, detectedNoise_array = \
                parser_one_mmw_demo_output_packet(allBinData[totalBytesParsed:], readNumBytes - totalBytesParsed, DEBUG)
        if DEBUG:
            print("Parser result: ", parser_result)
        if parser_result == 0:
            totalBytesParsed += headerStartIndex + totalPacketNumBytes
            numFramesParsed += 1
            detObj = {
                "numObj": numDetObj,
                "range": detectedRange_array,
                "x": detectedX_array,
                "y": detectedY_array,
                "z": detectedZ_array,
                "azimuth": detectedAzimuth_array,
                "snr": detectedSNR_array,
                "v": detectedV_array,
                "noise": detectedNoise_array,
            }
            dataOK = 1
        else:
            print("Error in parsing this frame; continue")
        shiftSize = totalPacketNumBytes
        byteBuffer[:byteBufferLength-shiftSize] = byteBuffer[shiftSize:byteBufferLength]
        byteBuffer[byteBufferLength-shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength-shiftSize:]), dtype='uint8')
        byteBufferLength -= shiftSize
        if byteBufferLength < 0:
            byteBufferLength = 0
    return dataOK, frameNumber, detObj

class SerialReader(QtCore.QThread):
    newData = QtCore.pyqtSignal(object, object, object, object, object)
    def run(self):
        global Dataport, configParameters
        while True:
            dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
            if dataOk and len(detObj.get("x", [])) > 0 and "azimuth" in detObj and "snr" in detObj and "v" in detObj:
                self.newData.emit(detObj["x"], detObj["y"], detObj["azimuth"], detObj["snr"], detObj["v"])
            self.msleep(10)

class MyWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.clusterCheckbox = QtWidgets.QCheckBox("Enable Clustering")
        self.clusterCheckbox.setChecked(False)
        self.clusterCheckbox.toggled.connect(self.onToggleClustering)
        self.mainLayout.addWidget(self.clusterCheckbox)
        self.enableClustering = False

        # New flag to limit to one (best) cluster.
        self.limit_to_best_cluster = True

        # Expanded default plot view to 10x10.
        self.plotItem = self.addPlot(title="Wall-mounted Sensor - User Detection")
        self.plotItem.enableAutoRange("xy", False)
        self.plotItem.setXRange(-5, 5)
        self.plotItem.setYRange(0, 10)
        self.plotItem.invertY(True)  # Flip vertical axis so that the top is closest.
        self.plotItem.setLabel("bottom", "Lateral Distance (m)")
        self.plotItem.setLabel("left", "Distance from Sensor (m)")

        self.rawDataItem = self.plotItem.plot([], pen=None, symbolBrush=(150,150,150),
                                                symbolSize=3, symbolPen=None)
        self.plotDataItem = self.plotItem.plot([], pen=None, symbolBrush=(0,255,0),
                                               symbolSize=10, symbolPen=None)

        self.textItems = []
        self.bboxItems = []
        self.last_bbox_params = []
        self.frameBuffer = []
        self.bufferSize = 10

        self.lastFilteredPoints = None
        self.lastFilteredAz = None
        self.lastFilteredSNR = None
        self.lastFilteredVelocity = None
        self.lastClusterCentroids = None
        self.updateCounter = 0
        self.clusteringInterval = 5

        # RIS confirmation variables.
        self.last_confirmed_AoA = None
        self.confirmation_count = 0
        self.confirmation_threshold = 2  # require 2 consecutive frames
        self.AoA_tolerance = 1.0  # tolerance in degrees

        self.grid_resolution = 0.1
        self.x_min = -2
        self.y_min = 0
        self.x_max = 2
        self.y_max = 5
        self.num_x = int((self.x_max - self.x_min) / self.grid_resolution)
        self.num_y = int((self.y_max - self.y_min) / self.grid_resolution)
        self.background_count = np.zeros((self.num_y, self.num_x))
        self.bg_decay = 0.95
        self.bg_threshold = 30

        self.lastBestCluster = None
        self.lastBestAoA = None
        self.missing_cluster_frames = 0
        self.max_missing_frames = 20
        self.dt = 0.1

        self.lastAoALabel = pg.TextItem("", color="w", anchor=(1,1))
        self.lastAoALabel.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
        self.plotItem.addItem(self.lastAoALabel, ignoreBounds=True)
        self.lastAoALabel.setPos(2, 5)

        self.kf = KalmanFilter(dt=0.1)
        self.kf_initialized = False

        self.serialThread = SerialReader()
        self.serialThread.newData.connect(self.updatePlot)
        self.serialThread.start()

    def onToggleClustering(self, checked):
        self.enableClustering = checked

    def updatePlot(self, x, y, az, snr, v):
        self.updateCounter += 1
        # Flip the horizontal axis: multiply x values by -1.
        newPoints = np.column_stack((x, y))
        newPoints = flip_x(newPoints)
        newAz = np.array(az)
        # Invert the AoA as well to match the x flip.
        newAz = -newAz
        newSNR = np.array(snr)
        newVelocity = np.array(v)

        if newPoints.size > 0:
            self.frameBuffer.append((newPoints, newAz, newSNR, newVelocity))
        if len(self.frameBuffer) > self.bufferSize:
            self.frameBuffer.pop(0)
        if len(self.frameBuffer) > 0:
            aggregatedPoints = np.vstack([fb[0] for fb in self.frameBuffer])
            aggregatedAz = np.hstack([fb[1] for fb in self.frameBuffer])
            aggregatedSNR = np.hstack([fb[2] for fb in self.frameBuffer])
            aggregatedV = np.hstack([fb[3] for fb in self.frameBuffer])
        else:
            aggregatedPoints = np.empty((0,2))
            aggregatedAz = np.empty((0,))
            aggregatedSNR = np.empty((0,))
            aggregatedV = np.empty((0,))

        snr_threshold = 10
        valid_idx = aggregatedSNR > snr_threshold
        filtered_points = aggregatedPoints[valid_idx]
        filtered_az = aggregatedAz[valid_idx]
        filtered_snr = aggregatedSNR[valid_idx]
        filtered_velocity = aggregatedV[valid_idx]

        if filtered_points.shape[0] == 0:
            if self.lastFilteredPoints is not None:
                filtered_points = self.lastFilteredPoints
                filtered_az = self.lastFilteredAz
                filtered_snr = self.lastFilteredSNR
                filtered_velocity = self.lastFilteredVelocity
            else:
                return
        else:
            self.lastFilteredPoints = filtered_points
            self.lastFilteredAz = filtered_az
            self.lastFilteredSNR = filtered_snr
            self.lastFilteredVelocity = filtered_velocity

        self.background_count *= self.bg_decay
        for pt in filtered_points:
            ix = int((pt[0] - self.x_min) / self.grid_resolution)
            iy = int((pt[1] - self.y_min) / self.grid_resolution)
            if 0 <= ix < self.num_x and 0 <= iy < self.num_y:
                self.background_count[iy, ix] += 1

        bg_mask = []
        for pt in filtered_points:
            ix = int((pt[0] - self.x_min) / self.grid_resolution)
            iy = int((pt[1] - self.y_min) / self.grid_resolution)
            if 0 <= ix < self.num_x and 0 <= iy < self.num_y:
                bg_mask.append(self.background_count[iy, ix] > self.bg_threshold)
            else:
                bg_mask.append(False)
        bg_mask = np.array(bg_mask)

        filtered_points_no_bg = filtered_points[~bg_mask]
        filtered_az_no_bg = filtered_az[~bg_mask]
        filtered_snr_no_bg = filtered_snr[~bg_mask]
        filtered_velocity_no_bg = filtered_velocity[~bg_mask]

        self.rawDataItem.setData(filtered_points[:, 0], filtered_points[:, 1])

        if self.enableClustering:
            if filtered_points_no_bg.shape[0] == 0:
                self.missing_cluster_frames += 1
                if self.missing_cluster_frames <= self.max_missing_frames and self.lastBestCluster is not None:
                    avg_velocity = np.mean(filtered_velocity, axis=0)
                    self.lastBestCluster = self.lastBestCluster + avg_velocity * self.dt
                    best_text = f"Az:{self.lastBestAoA:.1f}°"
                    self.plotDataItem.setData(np.array([self.lastBestCluster[0]]),
                                              np.array([self.lastBestCluster[1]]),
                                              symbolBrush=(0,255,0), symbolSize=10)
                    if self.last_bbox_params and not self.bboxItems:
                        for param in self.last_bbox_params:
                            rect_item = QGraphicsRectItem(param[0], param[1], param[2], param[3])
                            rect_item.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                            self.plotItem.addItem(rect_item)
                            self.bboxItems.append(rect_item)
                else:
                    self.plotDataItem.setData([], [])
                return

            base_eps = 0.2
            if filtered_snr_no_bg.size > 0:
                avg_snr = np.mean(filtered_snr_no_bg)
                eps = base_eps * (avg_snr / 20) if avg_snr < 20 else base_eps
                eps = max(eps, 0.1)
            else:
                eps = base_eps

            dbscan = DBSCAN(eps=eps, min_samples=4)
            labels = dbscan.fit_predict(filtered_points_no_bg)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            centroids = []
            cluster_azimuth = []
            cluster_sizes = []
            cluster_velocities = []
            valid_cluster_points = []

            static_velocity_threshold = 0.05
            static_cluster_size_threshold = 10

            for label in unique_labels:
                idx = (labels == label)
                cluster_points = filtered_points_no_bg[idx]
                cluster_az = filtered_az_no_bg[idx]
                cluster_v = filtered_velocity_no_bg[idx]
                cluster_size = len(cluster_points)
                avg_speed = np.mean(np.abs(cluster_v))
                if avg_speed < static_velocity_threshold and cluster_size < static_cluster_size_threshold:
                    continue
                avg_az = np.mean(cluster_az)
                center = np.mean(cluster_points, axis=0)
                # Compute bounding box width and height.
                min_x = np.min(cluster_points[:, 0])
                max_x = np.max(cluster_points[:, 0])
                width = max_x - min_x
                min_y = np.min(cluster_points[:, 1])
                max_y = np.max(cluster_points[:, 1])
                height = max_y - min_y
                if width < 0.2 or height < 0.2:  # require minimum width and height of 20 cm
                    continue
                centroids.append(center)
                cluster_azimuth.append(avg_az)
                cluster_sizes.append(cluster_size)
                cluster_velocities.append(avg_speed)
                valid_cluster_points.append(cluster_points)

            if len(centroids) == 0 and filtered_points_no_bg.shape[0] > 0:
                best_cluster_centroid = np.mean(filtered_points_no_bg, axis=0)
                if np.linalg.norm(best_cluster_centroid) <= 3.0:
                    best_cluster_az = np.mean(filtered_az_no_bg)
                    best_cluster_size = filtered_points_no_bg.shape[0]
                    best_cluster_velocity = np.mean(np.abs(filtered_velocity_no_bg))
                    centroids = [best_cluster_centroid]
                    cluster_azimuth = [best_cluster_az]
                    cluster_sizes = [best_cluster_size]
                    cluster_velocities = [best_cluster_velocity]
                    valid_cluster_points.append(filtered_points_no_bg)
                else:
                    centroids = []
                    cluster_azimuth = []
                    cluster_sizes = []
                    cluster_velocities = []
                    valid_cluster_points = []

            if len(centroids) > 0:
                if self.limit_to_best_cluster:
                    best_index = np.argmax(cluster_sizes)
                    best_cluster_points = valid_cluster_points[best_index]
                    best_cluster_centroid = np.array(centroids)[best_index]
                    best_cluster_az = cluster_azimuth[best_index]
                    
                    # --- Kalman filter integration with vertical movement threshold ---
                    vertical_threshold = 0.1  # 10 cm threshold for vertical movement
                    if not self.kf_initialized:
                        self.kf.x[0, 0] = best_cluster_centroid[0]
                        self.kf.x[1, 0] = best_cluster_centroid[1]
                        self.kf.x[4, 0] = best_cluster_az
                        self.kf_initialized = True
                    else:
                        if self.lastBestCluster is not None:
                            vertical_diff = abs(best_cluster_centroid[1] - self.lastBestCluster[1])
                            if vertical_diff < vertical_threshold:
                                best_cluster_centroid = self.lastBestCluster
                                best_cluster_az = self.lastBestAoA
                            else:
                                z = np.array([best_cluster_centroid[0], best_cluster_centroid[1], best_cluster_az])
                                self.kf.update(z)
                                pred_state = self.kf.predict()
                                best_cluster_centroid = np.array([pred_state[0, 0], pred_state[1, 0]])
                                best_cluster_az = pred_state[4, 0]
                        else:
                            z = np.array([best_cluster_centroid[0], best_cluster_centroid[1], best_cluster_az])
                            self.kf.update(z)
                            pred_state = self.kf.predict()
                            best_cluster_centroid = np.array([pred_state[0, 0], pred_state[1, 0]])
                            best_cluster_az = pred_state[4, 0]
                    # -------------------------------------------------------------
                    
                    # Use the bounding box from the DBSCAN cluster.
                    min_x = np.min(best_cluster_points[:, 0])
                    max_x = np.max(best_cluster_points[:, 0])
                    min_y = np.min(best_cluster_points[:, 1])
                    max_y = np.max(best_cluster_points[:, 1])
                    width = max_x - min_x
                    height = max_y - min_y

                    self.missing_cluster_frames = 0
                    self.lastBestCluster = best_cluster_centroid
                    self.lastBestAoA = best_cluster_az

                    self.plotDataItem.setData(np.array([best_cluster_centroid[0]]),
                                              np.array([best_cluster_centroid[1]]),
                                              symbolBrush=(0,255,0), symbolSize=10)
                    for textItem in self.textItems:
                        self.plotItem.removeItem(textItem)
                    self.textItems = []
                    best_textItem = pg.TextItem(f"Az:{best_cluster_az:.1f}°", color=(255,255,255))
                    best_textItem.setPos(best_cluster_centroid[0], best_cluster_centroid[1])
                    self.plotItem.addItem(best_textItem)
                    self.textItems.append(best_textItem)

                    for bbox in self.bboxItems:
                        self.plotItem.removeItem(bbox)
                    self.bboxItems = []
                    self.last_bbox_params = (min_x, min_y, width, height)
                    rect_item = QGraphicsRectItem(min_x, min_y, width, height)
                    rect_item.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                    self.plotItem.addItem(rect_item)
                    self.bboxItems.append(rect_item)
                    
                    # --- RIS Code Group ---
                    if self.last_confirmed_AoA is None:
                        self.last_confirmed_AoA = best_cluster_az
                        self.confirmation_count = 1
                    else:
                        if abs(best_cluster_az - self.last_confirmed_AoA) < self.AoA_tolerance:
                            self.confirmation_count += 1
                        else:
                            self.confirmation_count = 0
                            self.last_confirmed_AoA = best_cluster_az
                    if self.confirmation_count >= self.confirmation_threshold:
                        image_index = find_closest_image(-best_cluster_az) #ANGLE INVERT RIS
                        send_image_to_serial(image_index)
                        print(f"RIS configuration sent for AoA {best_cluster_az:.1f}° (Image {image_index})")
                        self.confirmation_count = 0
                    # -----------------------
                    
                else:
                    self.missing_cluster_frames = 0
                    for bbox in self.bboxItems:
                        self.plotItem.removeItem(bbox)
                    self.bboxItems = []
                    self.last_bbox_params = []
                    for pts, az in zip(valid_cluster_points, cluster_azimuth):
                        if pts.shape[0] > 0:
                            min_x = np.min(pts[:, 0])
                            max_x = np.max(pts[:, 0])
                            min_y = np.min(pts[:, 1])
                            max_y = np.max(pts[:, 1])
                            width = max_x - min_x
                            height = max_y - min_y
                            self.last_bbox_params.append((min_x, min_y, width, height))
                            rect_item = QGraphicsRectItem(min_x, min_y, width, height)
                            rect_item.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                            self.plotItem.addItem(rect_item)
                            self.bboxItems.append(rect_item)
                    centroids_arr = np.array(centroids)
                    self.plotDataItem.setData(centroids_arr[:,0], centroids_arr[:,1],
                                              symbolBrush=(0,255,0), symbolSize=10)
            else:
                self.missing_cluster_frames += 1
                if self.missing_cluster_frames <= self.max_missing_frames and self.lastBestCluster is not None:
                    avg_velocity = np.mean(filtered_velocity, axis=0)
                    self.lastBestCluster = self.lastBestCluster + avg_velocity * self.dt
                    best_text = f"Az:{self.lastBestAoA:.1f}°"
                    self.plotDataItem.setData(np.array([self.lastBestCluster[0]]),
                                              np.array([self.lastBestCluster[1]]),
                                              symbolBrush=(0,255,0), symbolSize=10)
                    if self.last_bbox_params and not self.bboxItems:
                        if isinstance(self.last_bbox_params, tuple):
                            rect_item = QGraphicsRectItem(self.last_bbox_params[0],
                                                          self.last_bbox_params[1],
                                                          self.last_bbox_params[2],
                                                          self.last_bbox_params[3])
                            rect_item.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                            self.plotItem.addItem(rect_item)
                            self.bboxItems.append(rect_item)
                        else:
                            for param in self.last_bbox_params:
                                rect_item = QGraphicsRectItem(param[0], param[1], param[2], param[3])
                                rect_item.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                                self.plotItem.addItem(rect_item)
                                self.bboxItems.append(rect_item)
                else:
                    self.plotDataItem.setData([], [])
        else:
            self.plotDataItem.setData(filtered_points_no_bg[:,0],
                                      filtered_points_no_bg[:,1],
                                      symbolBrush=(255,0,0), symbolSize=5)

        if self.lastBestAoA is not None:
            self.lastAoALabel.setText(f"AoA: {self.lastBestAoA:.1f}°")

    def closeEvent(self, event):
        self.serialThread.quit()
        self.serialThread.wait()
        CLIport.write(("sensorStop\n").encode())
        CLIport.close()
        Dataport.close()
        event.accept()

def main():
    global configParameters, Dataport
    CLIport, Dataport = serialConfig()
    configParameters = parseConfigFile(None)
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=False)
    win = MyWidget()
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()
    CLIport.write(("sensorStop\n").encode())
    CLIport.close()
    Dataport.close()

if __name__ == "__main__":
    main()
