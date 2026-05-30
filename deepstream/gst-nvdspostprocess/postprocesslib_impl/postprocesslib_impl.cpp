/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "postprocesslib_impl.h"
#include "nvdspreprocess_meta.h"
#include <filesystem>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// #include "thermal_render.h"

using namespace std;

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

extern "C" IDSPostProcessLibrary *CreateCustomAlgoCtx(DSPostProcess_CreateParams *params)
{
  return new PostProcessAlgorithm(params);
}


/*Separate a config file entry with delimiters
 *to be able to parse it.*/
std::vector<std::string>
PostProcessAlgorithm::SplitString (std::string input) {
 std::stringstream longStr(input);
  std::string item;
  std::vector <std::string> ret;
  while (std::getline (longStr, item, ';')){
    ret.push_back(item);
  }
  return ret;
}

std::set<gint>
PostProcessAlgorithm::SplitStringInt (std::string input) {

  std::stringstream longStr(input);
  std::string item;
  std::set <gint> ret;
  while (std::getline (longStr, item, ';')){
    ret.insert(stoi(item));
  }
  return ret;
}

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */

bool
PostProcessAlgorithm::GetAbsFilePath (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
        return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
bool PostProcessAlgorithm::ParseLabelsFile(
    const std::string& labelsFilePath) {
    std::ifstream labels_file(labelsFilePath, std::ios_base::in);
    std::string delim{';'};
    if (!labels_file) {
        printError("Could not open labels file:%s", safeStr(labelsFilePath));
        return false;
    }
    while (labels_file.good() && !labels_file.eof()) {
        std::string line, word;
        std::vector<std::string> l;
        size_t pos = 0, oldpos = 0;

        std::getline(labels_file, line, '\n');
        if (line.empty())
            continue;

        while ((pos = line.find(delim, oldpos)) != std::string::npos) {
            word = line.substr(oldpos, pos - oldpos);
            l.push_back(word);
            oldpos = pos + delim.length();
        }
        l.push_back(line.substr(oldpos));
        m_Labels.push_back(l);
    }

    if (labels_file.bad()) {
        printError("Failed to parse labels file:%s, iostate:%d",
            safeStr(labelsFilePath), (int)labels_file.rdstate());
        return false;
    }
    return true;
}


bool PostProcessAlgorithm::SetConfigFile (const gchar *cfg_file_path){
  bool ret = true;
  NvDsPostProcessStatus status = NVDSPOSTPROCESS_SUCCESS;

  if (!cfg_file_path || !std::filesystem::exists(cfg_file_path))
  {
    printError("Config File input not provided or doesn't exist");
    return false;
  }
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if (!(configyml.size() > 0)) {
    printError("Unable to parse config file '%s' ", cfg_file_path);
    return false;
  }

  m_processLock.lock();
  // Parse the config file here
  if(configyml["property"]) {
   for(YAML::const_iterator itr = configyml["property"].begin();
        itr != configyml["property"].end(); ++itr){
      std::string paramKey = itr->first.as<std::string>();
      if (paramKey == "gpu-id"){
        m_gpuId = itr->second.as<gint>();
        m_initParams.gpuID = m_gpuId;
      }
      else if (paramKey == "preprocessor-support"){
        m_preprocessor_support =  itr->second.as<gboolean>();
        m_initParams.preprocessor_support = m_preprocessor_support;
      }
      else if (paramKey == "network-type"){
        switch (itr->second.as<gint>()) {
          case NvDsPostProcessNetworkType_Detector:
          case NvDsPostProcessNetworkType_Classifier:
          case NvDsPostProcessNetworkType_Segmentation:
          case NvDsPostProcessNetworkType_InstanceSegmentation:
          case NvDsPostProcessNetworkType_BodyPose:
          case NvDsPostProcessNetworkType_Other:
            m_networkType = static_cast<NvDsPostProcessNetworkType>(itr->second.as<gint>());
            break;
          default:
            g_printerr ("Error. Invalid value for 'network-type':'%d'\n",
                itr->second.as<gint>());
            return false;
            break;
        }
        m_initParams.networkType = m_networkType;
      }
      else if (paramKey == "process-mode"){
        m_processMode = itr->second.as<gint>();
      }
      else if (paramKey == "num-detected-classes"){
        m_numDetectedClasses = itr->second.as<gint>();
        m_initParams.numDetectedClasses = m_numDetectedClasses;
      }
      else if (paramKey == "gie-unique-id"){
        m_gieUniqueId = itr->second.as<gint>();
        m_initParams.uniqueID = m_gieUniqueId;
      }
      else if (paramKey == "labelfile-path"){
        ret = ParseLabelsFile(itr->second.as<std::string>());
        if (ret ==false){
          return ret;
        }
        std::strncpy(m_initParams.labelsFilePath,
            (itr->second.as<std::string>()).c_str(), sizeof(m_initParams.labelsFilePath)-1);
      }
      else if (paramKey == "cluster-mode"){
        switch (itr->second.as<gint>())
        {
          case 0:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES;
            break;
          case 1:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_DBSCAN;
            break;
          case 2:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_NMS;
            break;
          case 3:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_DBSCAN_NMS_HYBRID;
            break;
          case 4:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_NONE;
            break;
          default:
            g_printerr ("Error. Invalid value for 'cluster-mode':'%d'\n",
                itr->second.as<gint>());
            return false;
            break;
        }
        m_initParams.clusterMode = m_clusterMode;
      }
      else if (paramKey == "release-tensor-meta"){
        m_releaseTensorMeta = itr->second.as<gboolean>();
      }
      else if (paramKey == "output-instance-mask"){
        m_outputInstanceMask = itr->second.as<gboolean>();
      }
      else if (paramKey == "is-classifier"){
        m_isClassifier = itr->second.as<gboolean>();
      }
      else if (paramKey == "classifier-threshold"){
        m_classifierThreshold = itr->second.as<gfloat>();
        m_initParams.classifierThreshold = m_classifierThreshold;
      }
      else if (paramKey == "classifier-type"){
        m_classifierType = itr->second.as<std::string>();
        g_free (m_initParams.classifier_type);
        m_initParams.classifier_type = g_strdup(m_classifierType.c_str());
      }
      else if (paramKey == "segmentation-threshold"){
        m_segmentationThreshold = itr->second.as<gfloat>();
        m_initParams.segmentationThreshold = m_segmentationThreshold;
      }
      else if (paramKey == "segmentation-output-order"){
        m_initParams.segmentationOutputOrder =
          static_cast<NvDsPostProcessTensorOrder>(itr->second.as<gint>());
      }
      else if (paramKey == "parse-classifier-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customClassifierParseFuncName, temp.c_str(),
            sizeof(m_initParams.customClassifierParseFuncName)-1);
      }
      else if (paramKey == "parse-bbox-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customBBoxParseFuncName, temp.c_str(),
            sizeof(m_initParams.customBBoxParseFuncName)-1);
      } else if (paramKey == "parse-bbox-instance-mask-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customBBoxInstanceMaskParseFuncName, temp.c_str(),
            sizeof(m_initParams.customBBoxInstanceMaskParseFuncName)-1);
      }
      else if (paramKey == "output-blob-names"){
        m_outputBlobNames = SplitString (itr->second.as<std::string>());
        gchar **values;
        int len = (int) m_outputBlobNames.size();
        if (m_initParams.outputLayerNames){
          for (guint i=0; i < m_initParams.numOutputLayers; i++){
            g_free(m_initParams.outputLayerNames[i]);
            m_initParams.outputLayerNames[i] = NULL;
          }
          g_free (m_initParams.outputLayerNames);
        }
        values = g_new (gchar *, len + 1);
        for (int i = 0; i < len; i++) {
          int size = 64;
          char* str2 = (char*) g_malloc0(sizeof(char) * size);
          std::strncpy (str2, m_outputBlobNames[i].c_str(), size-1);
          values[i] = str2;
        }
        values[len] = NULL;
        m_initParams.outputLayerNames = values;
        m_initParams.numOutputLayers = len;
      }
      else if (paramKey == "operate-on-class-ids"){
        m_operateOnClassIds = SplitStringInt (itr->second.as<std::string>());
      }
      else if (paramKey == "filter-out-class-ids"){
        m_filterOutClassIds = SplitStringInt (itr->second.as<std::string>());
      }
      else {
        printWarning ("Unknown parameter %s ",paramKey.c_str());
      }
   }
  }
  else {
    printError("property group not present in config file '%s' ", cfg_file_path);
    m_processLock.unlock();
    return false;
  }

  if (m_initParams.networkType == NvDsPostProcessNetworkType_Detector ||
      m_initParams.networkType == NvDsPostProcessNetworkType_InstanceSegmentation){
    NvDsPostProcessDetectionParams detection_params{DEFAULT_PRE_CLUSTER_THRESHOLD,
        DEFAULT_POST_CLUSTER_THRESHOLD, DEFAULT_EPS,
        DEFAULT_GROUP_THRESHOLD, DEFAULT_MIN_BOXES,
        DEFAULT_DBSCAN_MIN_SCORE, DEFAULT_NMS_IOU_THRESHOLD, DEFAULT_TOP_K,
        0, 0, 0, 0, 0, 0,
        {TRUE, (NvOSD_ColorParams){1.0,0.0,0.0,1.0},
         FALSE,(NvOSD_ColorParams){1.0,0.0,0.0,1.0}}};
    detection_params.color_params.have_border_color = TRUE;
    detection_params.color_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
    detection_params.color_params.have_bg_color = FALSE;

    /* Parse the parameters for "all" classes if the group has been specified.
     * Detection/Segmentation */
    if (configyml["class-attrs-all"]) {
      ret = ParseConfAttr (configyml["class-attrs-all"], -1, detection_params);
      if (ret ==false){
        printError("Parsing 'class-attrs-all' group failed");
        return ret;
      }
    }

    /* Initialize the per-class vector with the same default/parsed values for
     * all classes. */
    if (m_initParams.perClassDetectionParams){
      delete [] m_initParams.perClassDetectionParams;
    }
    m_initParams.perClassDetectionParams =
      new NvDsPostProcessDetectionParams[m_initParams.numDetectedClasses];

    for (uint32_t icnt = 0; icnt < m_initParams.numDetectedClasses; icnt++){
      m_initParams.perClassDetectionParams[icnt] = detection_params;
    }

    for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      std::string class_str = "class-attrs-";
      if ((paramKey != "class-attrs-all") &&
          (paramKey.size() >= class_str.size()))  {
        if (class_str.compare(0,class_str.size(),paramKey.c_str(),
              class_str.size()) == 0) {
          std::string num_str = paramKey.substr(class_str.size());
          gint64 class_index = stoi(num_str);
          m_initParams.perClassDetectionParams[class_index] = detection_params;
          ret = ParseConfAttr (configyml[paramKey], class_index,
              m_initParams.perClassDetectionParams[class_index]);
          if (ret ==false){
            printError("Parsing '%s' group failed",paramKey.c_str());
            return ret;
          }
        }
      }
    }
    status = preparePostProcess();
    if (status != NVDSPOSTPROCESS_SUCCESS){
      return false;
    }
  }
  else if (m_initParams.networkType == NvDsPostProcessNetworkType_Classifier ||
          m_initParams.networkType == NvDsPostProcessNetworkType_Segmentation ||
          m_initParams.networkType == NvDsPostProcessNetworkType_BodyPose ||
          m_initParams.networkType == 100) {

      status = preparePostProcess();
      if (status != NVDSPOSTPROCESS_SUCCESS){
        return false;
      }
  }
  else {
      printError("Parsing for network type %d is not supported",
          m_initParams.networkType);
      return false;
  }

  m_processLock.unlock();
  return true;
}

NvDsPostProcessStatus
PostProcessAlgorithm::preparePostProcess(){
  NvDsPostProcessStatus ret = NVDSPOSTPROCESS_CONFIG_FAILED;

  m_Postprocessor.reset();
  switch (m_initParams.networkType){
  case NvDsPostProcessNetworkType_Detector:
    m_Postprocessor = std::make_unique<DetectModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
  case NvDsPostProcessNetworkType_Classifier:
    m_Postprocessor = std::make_unique<ClassifyModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
  case NvDsPostProcessNetworkType_Segmentation:
    m_Postprocessor = std::make_unique<SegmentationModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
  case NvDsPostProcessNetworkType_InstanceSegmentation:
    m_Postprocessor = std::make_unique<InstanceSegmentModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
  case NvDsPostProcessNetworkType_BodyPose:
    m_Postprocessor = std::make_unique<BodyPoseModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
    //FIXME:
  case NvDsPostProcessNetworkType_Other:
    m_Postprocessor = nullptr;
    return NVDSPOSTPROCESS_SUCCESS;
  default:
    printError(" Failed to validate the network type, unknown network %d",m_initParams.networkType);
    return ret;
  }
  if (ret != NVDSPOSTPROCESS_SUCCESS){
    m_Postprocessor.reset();
  }
  return ret;
}

bool PostProcessAlgorithm::ParseConfAttr (YAML::Node node, gint64 class_index,
    NvDsPostProcessDetectionParams& params)
{
  bool ret = true;

  for(YAML::const_iterator itr = node.begin(); itr != node.end(); ++itr) {

    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "detected-min-w"){
      params.detectionMinWidth = itr->second.as<gint>();
    }
    else if (paramKey == "detected-min-h"){
      params.detectionMinHeight = itr->second.as<gint>();
    }
    else if (paramKey == "detected-max-w"){
      params.detectionMaxWidth = itr->second.as<gint>();
    }
    else if (paramKey == "detected-max-h"){
      params.detectionMaxHeight = itr->second.as<gint>();
    }
    else if (paramKey == "minBoxes"){
      params.minBoxes = itr->second.as<gint>();
    }
    else if (paramKey == "pre-cluster-threshold"){
      params.preClusterThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "post-cluster-threshold"){
      params.postClusterThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "eps"){
      params.eps = itr->second.as<gfloat>();
    }
    else if (paramKey == "group-threshold"){
      params.groupThreshold = itr->second.as<gint>();
    }
    else if (paramKey == "min-score"){
      params.minScore = itr->second.as<gfloat>();
    }
    else if (paramKey == "dbscan-min-score"){
      params.minScore = itr->second.as<gfloat>();
    }
    else if (paramKey == "nms-iou-threshold"){
      params.nmsIOUThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "topk"){
      params.topK = itr->second.as<gint>();
    }
    else if (paramKey == "roi-top-offset"){
      params.roiTopOffset = itr->second.as<gint>();
    }
    else if (paramKey == "roi-bottom-offset"){
      params.roiBottomOffset = itr->second.as<gint>();
    }
    else if (paramKey == "border-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = SplitString(values);
      if (vec.size() != 4){
         g_printerr
            ("Error: in border-color, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
         ret = false;
        goto done;
      }
      params.color_params.border_color.red = std::stod(vec[0]);
      params.color_params.border_color.green = std::stod(vec[1]);
      params.color_params.border_color.blue = std::stod(vec[2]);
      params.color_params.border_color.alpha = std::stod(vec[3]);
    }
    else if (paramKey == "bg-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = SplitString(values);

      if (vec.size() != 4) {
        g_printerr
            ("Error: Group bg-color, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        ret = false;
        goto done;
      }
      params.color_params.bg_color.red = std::stod(vec[0]);
      params.color_params.bg_color.green = std::stod(vec[1]);
      params.color_params.bg_color.blue = std::stod(vec[2]);
      params.color_params.bg_color.alpha = std::stod(vec[3]);
      params.color_params.have_bg_color = TRUE;
    }
    else {
      printWarning ("Unknown parameter '%s' ",paramKey.c_str());
    }
  }
  m_detectorClassAttr[class_index] = params;
done:
  return ret;
}


bool PostProcessAlgorithm::HandleEvent (GstEvent *event)
{
  switch (GST_EVENT_TYPE(event))
  {
       case GST_EVENT_EOS:
           m_processLock.lock();
           m_stop = TRUE;
           m_processCV.notify_all();
           m_processLock.unlock();
           while (outputthread_stopped == FALSE)
           {
               //g_print ("waiting for processq to be empty, buffers in processq = %ld\n", m_processQ.size());
               g_usleep (1000);
           }
           break;
       default:
           break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED)
  {
      gst_nvevent_parse_pad_added (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
  {
      gst_nvevent_parse_pad_deleted (event, &source_id);
  }
  return true;
}


/* Deinitialize the Custom Lib context */
PostProcessAlgorithm::~PostProcessAlgorithm()
{
  std::unique_lock<std::mutex> lk(m_processLock);
  m_processCV.wait(lk, [&]{return m_processQ.empty();});
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }

  if (m_initParams.perClassDetectionParams){
    delete[] m_initParams.perClassDetectionParams;
  }
  if (m_initParams.outputLayerNames){
    for (uint32_t i = 0; i < m_initParams.numOutputLayers; i++){
      g_free(m_initParams.outputLayerNames[i]);
      m_initParams.outputLayerNames[i] = NULL;
    }
    g_free (m_initParams.outputLayerNames);
  }

  // ThermalRender_Shutdown();
}

// Returns NvDsBatchMeta if present in the gstreamer buffer else NULL
NvDsBatchMeta *PostProcessAlgorithm::GetNVDS_BatchMeta (GstBuffer *buffer)
{
  gpointer state = NULL;
  GstMeta *gst_meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta(buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      continue;
    }
    NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      if (batch_meta != NULL) {
        GST_WARNING("Multiple NvDsBatchMeta found on buffer %p", buffer);
      }
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
    }
  }
  return batch_meta;
}


/* Process Buffer */
BufferResult PostProcessAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GstMapInfo in_map_info = GST_MAP_INFO_INIT;

  GST_DEBUG_OBJECT (m_element, "PostProcessLib: ---> Inside %s frame_num = %d\n", __func__, m_frameNum++);

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return BufferResult::Buffer_Error;
  }
  gst_buffer_unmap(inbuf, &in_map_info);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;

  // Add custom preprocessing logic if required, here
  // Pass the buffer to output_loop for further processing and pushing to next component

  // Enable for dumping the input frame, for debugging purpose
  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}


/* Output Processing Thread */
void PostProcessAlgorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  std::unique_lock<std::mutex> lk(m_processLock);
  NvDsBatchMeta *batch_meta = NULL;
  int32_t frame_cnt = 0;
  /* Run till signalled to stop. */
  while (1) {

    /* Wait if processing queue is empty. */
    if (m_processQ.empty()) {
      if (m_stop == TRUE) {
        break;
      }
      m_processCV.wait(lk);
      continue;
    }

    PacketInfo packetInfo = m_processQ.front();
    m_processQ.pop();

    m_processCV.notify_all();
    lk.unlock();

    // Add post process algorithm logic here
    // Once buffer processing is done, push the buffer to the downstream
    // by using gst_pad_push function

    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);
    batch_meta = GetNVDS_BatchMeta (packetInfo.inbuf);
    outBuffer = packetInfo.inbuf;
    nvds_set_input_system_timestamp (outBuffer, GST_ELEMENT_NAME(m_element));
    if(m_preprocessor_support)
    {
      for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
            l_user != NULL; l_user = l_user->next) {
          NvDsUserMeta *roi_user_meta = (NvDsUserMeta *) l_user->data;
          if (roi_user_meta->base_meta.meta_type != NVDS_ROI_META)
            continue;
          /* convert to roi metadata */
          NvDsRoiMeta *roi_meta =
            (NvDsRoiMeta *) roi_user_meta->user_meta_data;
          for (NvDsUserMetaList * r_user = roi_meta->roi_user_meta_list;
          r_user != NULL; r_user = r_user->next){
            NvDsUserMeta *tensor_user_meta = (NvDsUserMeta *) r_user->data;
            if (tensor_user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
              continue;
            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta =
              (NvDsInferTensorMeta *) tensor_user_meta->user_meta_data;
            /* PGIE and operate on meta->unique_id data only */
            if (meta->unique_id == m_gieUniqueId){
              for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
              }
              std::vector < NvDsInferLayerInfo >
                outputLayersInfo (meta->output_layers_info,
                    meta->output_layers_info + meta->num_output_layers);
              NvDsPostProcessFrameOutput output;
              memset (&output, 0, sizeof(output));
              if (m_Postprocessor){
                m_Postprocessor->setNetworkInfo(meta->network_info);
                m_Postprocessor->parseEachFrame(outputLayersInfo, output);
                m_Postprocessor->attachMetadata (in_surf, frame_meta->batch_id,
                    batch_meta, frame_meta, NULL, NULL,
                    output,
                    m_initParams.perClassDetectionParams,
                    m_filterOutClassIds,
                    m_gieUniqueId,
                    m_outputInstanceMask,
                    m_processMode, m_segmentationThreshold,
                    meta->maintain_aspect_ratio, roi_meta,
                    meta->symmetric_padding);
                m_Postprocessor->releaseFrameOutput (output);
              }
              else {
                GST_WARNING_OBJECT(m_element, "Post Processor not initialized for network");
              }
            }
          }
        }
      }
    }
    else
    {
        /* Iterate each frame metadata in batch */
        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
            l_frame != NULL;
            l_frame = l_frame->next)
        {

            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

            if (m_processMode == PROCESS_MODEL_FULL_FRAME)
            {                    
                /* 🔵 buscar OUTPUT tensor (incluye INPUT internamente) */
                // Con input-tensor-from-meta=1, el tensor está dentro del ROI meta
                for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list;
                    l_user != NULL; l_user = l_user->next)
                {
                    NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;

                    // Nivel 1: buscar ROI meta
                    if (user_meta->base_meta.meta_type != NVDS_ROI_META)
                        continue;

                    NvDsRoiMeta *roi_meta = (NvDsRoiMeta *)user_meta->user_meta_data;

                    // Nivel 2: buscar tensor meta dentro del ROI
                    for (NvDsUserMetaList *r_user = roi_meta->roi_user_meta_list;
                        r_user != NULL; r_user = r_user->next)
                    {
                        NvDsUserMeta *tensor_user_meta = (NvDsUserMeta *)r_user->data;

                        if (tensor_user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                            continue;

                        NvDsInferTensorMeta *meta =
                            (NvDsInferTensorMeta *)tensor_user_meta->user_meta_data;

                        if (meta->unique_id != m_gieUniqueId)
                            continue;

                        // Asignar host buffers
                        for (unsigned int i = 0; i < meta->num_output_layers; i++)
                            meta->output_layers_info[i].buffer = meta->out_buf_ptrs_host[i];

                        RenderThermalOutput(in_surf, frame_meta->batch_id, meta, frame_meta);
                    }
                }
            }
        }
    }

    nvds_set_output_system_timestamp (outBuffer, GST_ELEMENT_NAME(m_element));
    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element), outBuffer);
    GST_DEBUG_OBJECT (m_element,
    "CustomLib: %s in_surf=%p, Pushing Frame %d to downstream... Frame %d flow_ret = %d"\
    " TS=%" GST_TIME_FORMAT " \n",
            __func__, in_surf, packetInfo.frame_num, frame_cnt++,
            flow_ret,
            GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

    lk.lock();
    continue;
  }
  outputthread_stopped = true;
  lk.unlock();
  return;
}







// ============================================================
// RenderThermalOutput
// ============================================================

static constexpr float TS_MEAN  = 1861.5075235004497f;
static constexpr float TS_STD   = 296.8565934989852f;
static constexpr float T_MIN    = 1500.0f;
static constexpr float T_MAX    = 2150.0f;
static constexpr float SCALE_Z  = 5.5f  / 128.0f;
static constexpr float SCALE_R  = 0.6f  / 32.0f; 
static constexpr int   N_TICKS_Z = 12;
static constexpr int   N_TICKS_R = 3;

// Máscara Otsu dinámica
static cv::Mat compute_otsu_mask(const float* chw_data, int H, int W)
{
    const float* G = chw_data + H * W;

    float g_min = *std::min_element(G, G + H * W);
    float g_max = *std::max_element(G, G + H * W);

    cv::Mat G_mat(H, W, CV_32F);
    float range = g_max - g_min;
    for (int i = 0; i < H * W; i++)
        G_mat.at<float>(i / W, i % W) =
            (range > 0.f) ? (G[i] - g_min) / range : 0.f;

    cv::Mat G_uint8;
    G_mat.convertTo(G_uint8, CV_8U, 255.0);

    cv::Mat mask;
    double thresh = cv::threshold(G_uint8, mask, 0, 255,
                                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::cout << "[OTSU] threshold=" << thresh << std::endl;

    return mask;  // CV_8U, 255=llama 0=fondo
}

// Panel Ts + canales RGB
static cv::Mat build_ts_rgb_panel(
    const float* tensor_chw,
    const float* Ts,
    const cv::Mat& mask,
    int H, int W)
{
    // Ts → Kelvin → colormap
    cv::Mat Ts_k(H, W, CV_32F);
    for (int i = 0; i < H * W; i++)
        Ts_k.at<float>(i / W, i % W) = Ts[i] * TS_STD + TS_MEAN;

    cv::Mat Ts_norm(H, W, CV_32F);
    for (int i = 0; i < H * W; i++) {
        float v = (Ts_k.at<float>(i / W, i % W) - T_MIN) / (T_MAX - T_MIN);
        Ts_norm.at<float>(i / W, i % W) = std::max(0.f, std::min(1.f, v));
    }

    cv::Mat Ts_u8;
    Ts_norm.convertTo(Ts_u8, CV_8U, 255.0);
    cv::Mat Ts_color;
    cv::applyColorMap(Ts_u8, Ts_color, cv::COLORMAP_INFERNO);

    // Zona fuera de llama → blanco
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (!mask.at<uchar>(y, x))
                Ts_color.at<cv::Vec3b>(y, x) = {255, 255, 255};

    // Canales RGB
    std::vector<cv::Mat> rgb_panels;
    for (int c : {0, 1, 2}) {
        const float* ch_ptr = tensor_chw + c * H * W;
        float cmin = *std::min_element(ch_ptr, ch_ptr + H * W);
        float cmax = *std::max_element(ch_ptr, ch_ptr + H * W);

        cv::Mat ch_norm(H, W, CV_32F);
        for (int i = 0; i < H * W; i++) {
            float v = (cmax > cmin)
                ? (ch_ptr[i] - cmin) / (cmax - cmin)
                : 0.f;
            ch_norm.at<float>(i / W, i % W) = v;
        }

        cv::Mat ch_u8;
        ch_norm.convertTo(ch_u8, CV_8U, 255.0);
        cv::Mat ch_color;
        cv::applyColorMap(ch_u8, ch_color, cv::COLORMAP_VIRIDIS);

        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (!mask.at<uchar>(y, x))
                    ch_color.at<cv::Vec3b>(y, x) = {255, 255, 255};

        rgb_panels.push_back(ch_color);
    }

    // Panel horizontal: Ts | R | G | B
    cv::Mat row_panel;
    std::vector<cv::Mat> cols = {Ts_color,
                                  rgb_panels[0],
                                  rgb_panels[1],
                                  rgb_panels[2]};
    cv::hconcat(cols, row_panel);

    // Labels
    const cv::Scalar black(0, 0, 0);
    const cv::Scalar white(255, 255, 255);
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::putText(row_panel, "Ts",          {5,        14}, font, 0.4, black, 1);
    cv::putText(row_panel, "B",           {W + 5,    14}, font, 0.4, black, 1);
    cv::putText(row_panel, "G",           {2*W + 5,  14}, font, 0.4, black, 1);
    cv::putText(row_panel, "R",           {3*W + 5,  14}, font, 0.4, black, 1);

    // Temperatura media sobre llama
    float sum_T = 0.f; int cnt = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (mask.at<uchar>(y, x)) {
                sum_T += Ts_k.at<float>(y, x);
                cnt++;
            }
    if (cnt > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "Mean: %.0fK", sum_T / cnt);
        cv::putText(row_panel, buf, {5, H - 4}, font, 0.35, black, 1);
    }

    // Colorbars
    int cb_w = row_panel.cols;
    int cb_h = 16;
    cv::Mat grad(1, cb_w, CV_8U);
    for (int x = 0; x < cb_w; x++)
        grad.at<uchar>(0, x) = (uchar)(x * 255 / (cb_w - 1));

    cv::Mat cb_ts_1d, cb_rgb_1d;
    cv::applyColorMap(grad, cb_ts_1d,  cv::COLORMAP_INFERNO);
    cv::applyColorMap(grad, cb_rgb_1d, cv::COLORMAP_VIRIDIS);

    cv::Mat cb_ts  = cv::repeat(cb_ts_1d,  cb_h, 1);
    cv::Mat cb_rgb = cv::repeat(cb_rgb_1d, cb_h, 1);

    char buf_min[16], buf_max[16];
    std::snprintf(buf_min, sizeof(buf_min), "%.0f", T_MIN);
    std::snprintf(buf_max, sizeof(buf_max), "%.0f", T_MAX);
    cv::putText(cb_ts,  buf_min, {4, cb_h - 3}, font, 0.35, white, 1);
    cv::putText(cb_ts,  buf_max, {cb_w - 42, cb_h - 3}, font, 0.35, white, 1);
    cv::putText(cb_rgb, "0",     {4, cb_h - 3}, font, 0.35, white, 1);
    cv::putText(cb_rgb, "1",     {cb_w - 12, cb_h - 3}, font, 0.35, white, 1);

    cv::Mat final_panel;
    cv::vconcat(std::vector<cv::Mat>{row_panel, cb_ts, cb_rgb}, final_panel);
    return final_panel;
}

// Gráfico centerline
static cv::Mat build_centerline_panel(
    const float* Ts, const cv::Mat& mask, int H, int W)
{
    int col = W / 2;

    std::vector<int>   z_vals;
    std::vector<float> T_vals;

    for (int y = 0; y < H; y++) {
        if (!mask.at<uchar>(y, col)) continue;
        float T_k = Ts[y * W + col] * TS_STD + TS_MEAN;
        z_vals.push_back(y);
        T_vals.push_back(T_k);
    }

    const int W_c = 300, H_c = 200, M = 35;
    cv::Mat canvas(H_c, W_c, CV_8UC3, cv::Scalar(255, 255, 255));

    if (z_vals.empty()) {
        cv::putText(canvas, "No data", {W_c/2 - 30, H_c/2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1);
        return canvas;
    }

    // Escalas
    float z_min = (float)z_vals.front();
    float z_max = (float)z_vals.back();

    auto px_x = [&](float z) -> int {
        return M + (int)((z - z_min) / (z_max - z_min + 1e-8f)
                         * (W_c - 2 * M));
    };
    auto px_y = [&](float T) -> int {
        return H_c - M - (int)(((T - T_MIN) / (T_MAX - T_MIN + 1e-8f))
                               * (H_c - 2 * M));
    };

    // Ejes
    cv::line(canvas, {M, H_c - M}, {W_c - M, H_c - M}, {0,0,0}, 1);
    cv::line(canvas, {M, M},       {M, H_c - M},        {0,0,0}, 1);

    // Ticks eje Z
    for (int t = 0; t <= N_TICKS_Z; t++) {
        float z_t = z_min + t * (z_max - z_min) / N_TICKS_Z;
        int   px  = px_x(z_t);
        cv::line(canvas, {px, H_c - M}, {px, H_c - M + 3}, {0,0,0}, 1);
        if (t % 3 == 0) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%.1f", z_t * SCALE_Z);
            cv::putText(canvas, buf, {px - 10, H_c - M + 12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, {0,0,0}, 1);
        }
    }

    // Ticks eje T
    for (int t = 0; t <= N_TICKS_R; t++) {
        float T_t = T_MIN + t * (T_MAX - T_MIN) / N_TICKS_R;
        int   py  = px_y(T_t);
        cv::line(canvas, {M - 3, py}, {M, py}, {0,0,0}, 1);
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%.0f", T_t);
        cv::putText(canvas, buf, {1, py + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.28, {0,0,0}, 1);
    }

    // Curva
    for (int i = 0; i + 1 < (int)z_vals.size(); i++)
        cv::line(canvas,
                 {px_x((float)z_vals[i]),     px_y(T_vals[i])},
                 {px_x((float)z_vals[i + 1]), px_y(T_vals[i + 1])},
                 {0, 0, 200}, 2);

    // Labels ejes
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::putText(canvas, "z [mm]", {W_c/2 - 20, H_c - 3},  font, 0.35, {0,0,0}, 1);
    cv::putText(canvas, "T[K]",   {1, M - 5},               font, 0.35, {0,0,0}, 1);

    return canvas;
}

// RenderThermalOutput
void PostProcessAlgorithm::RenderThermalOutput(
    NvBufSurface* surf,
    guint         batch_id,
    NvDsInferTensorMeta* meta,
    NvDsFrameMeta* frame_meta)
{
    // Guards
    if (!meta || meta->num_output_layers == 0) return;

    auto& layer = meta->output_layers_info[0];

    // Control de frecuencia de visualización 
    // Cambiar RENDER_EVERY para visualizar cada N frames
    static constexpr int RENDER_EVERY = 1;
    static std::atomic<int> frame_counter{0};
    int cur = frame_counter.fetch_add(1);
    if (cur % RENDER_EVERY != 0) return;

    // Parseo de dims
    int C = 1, H = 128, W = 32;
    if (layer.inferDims.numDims == 3) {
        C = layer.inferDims.d[0];
        H = layer.inferDims.d[1];
        W = layer.inferDims.d[2];
    } else if (layer.inferDims.numDims == 4) {
        C = layer.inferDims.d[1];
        H = layer.inferDims.d[2];
        W = layer.inferDims.d[3];
    }
    int S = C * H * W;

    // Copiar tensor output a host
    size_t batch_off = (layer.inferDims.numDims == 4) ? batch_id : 0;
    std::vector<float> Ts(S);

    if (layer.dataType == NvDsInferDataType::FLOAT) {
        const float* src =
            reinterpret_cast<float*>(layer.buffer) + batch_off * S;
        std::memcpy(Ts.data(), src, S * sizeof(float));
    } else if (layer.dataType == NvDsInferDataType::HALF) {
        const __half* src =
            reinterpret_cast<__half*>(layer.buffer) + batch_off * S;
        for (int i = 0; i < S; i++)
            Ts[i] = __half2float(src[i]);
    } else {
        std::cout << "RenderThermalOutput: unsupported datatype" << std::endl;
        return;
    }

    // Obtener tensor de entrada (CHW float32)
    // Buscar en batch_user_meta_list el tensor de preprocesamiento
    std::vector<float> input_tensor(3 * H * W, 0.f);
    bool got_input = false;

    if (frame_meta && frame_meta->base_meta.batch_meta) {
        NvDsBatchMeta* bm = frame_meta->base_meta.batch_meta;
        for (NvDsMetaList* l = bm->batch_user_meta_list;
             l != nullptr; l = l->next)
        {
            auto* um = reinterpret_cast<NvDsUserMeta*>(l->data);
            if (um->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
                continue;
            auto* pm = reinterpret_cast<GstNvDsPreProcessBatchMeta*>(um->user_meta_data);
            if (!pm->tensor_meta) continue;

            // raw_tensor_buffer es float32 CHW en device (unified memory)
            const float* dev_ptr =
                reinterpret_cast<const float*>(
                    pm->tensor_meta->raw_tensor_buffer);

            // Copiar a host 
            // Unified memory → accesible directamente,
            // pero cudaMemcpy garantiza coherencia
            cudaMemcpy(input_tensor.data(), dev_ptr,
                       3 * H * W * sizeof(float),
                       cudaMemcpyDeviceToHost);
            got_input = true;
            break;
        }
    }

    // Máscara Otsu 
    cv::Mat mask;
    if (got_input) {
        mask = compute_otsu_mask(input_tensor.data(), H, W);
    } else {
        // Fallback: máscara completa
        mask = cv::Mat(H, W, CV_8U, cv::Scalar(255));
        std::cout << "[OTSU] input tensor not available, using full mask" << std::endl;
    }

    // Construir paneles
    cv::Mat panel_ts = build_ts_rgb_panel(
        got_input ? input_tensor.data() : nullptr,
        Ts.data(), mask, H, W);

    cv::Mat panel_cl = build_centerline_panel(Ts.data(), mask, H, W);

    // Igualar altura
    if (panel_cl.rows != panel_ts.rows)
        cv::resize(panel_cl, panel_cl, {panel_cl.cols, panel_ts.rows});

    cv::Mat combined;
    cv::hconcat(panel_ts, panel_cl, combined);

    // TEMPORAL: Guardar en disco
    // Nombrar por número de frame para tener una imagen por frame
    static std::atomic<int> save_counter{0};
    int frame_id = save_counter.fetch_add(1);

    std::string fname = "render_frame_" + std::to_string(frame_id) + ".png";
    bool ok = cv::imwrite(fname, combined);
    std::cout << "[RENDER] saved " << fname
              << " (" << combined.cols << "x" << combined.rows << ")"
              << (ok ? " OK" : " FAILED") << std::endl;
    std::cout.flush();
}

