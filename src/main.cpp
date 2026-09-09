#include <vulkan/vulkan.h>
#include "quant.h"
#include "reference.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("VK error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

static const int NBIND = 6;

struct Offsets {
    VkDeviceSize a0, x0, x1, outF, w[5], bq[5], mult[5], total;
};

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkBuffer bigbuf = VK_NULL_HANDLE;
    VkDeviceMemory devmem = VK_NULL_HANDLE;
    VkBuffer stgbuf = VK_NULL_HANDLE;
    VkDeviceMemory stgmem = VK_NULL_HANDLE;
    void* stgmapped = nullptr;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipes[5] = {};
    VkPipeline smokepipe = VK_NULL_HANDLE;
    VkPipelineLayout smokelayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout smokedsl = VK_NULL_HANDLE;
    VkDescriptorSet smokedb = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    float tsPeriod = 1.0f;
    int scope = -1;
    Offsets off{};
};

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return 0xFFFFFFFF;
}

static VkShaderModule load_spv(VkDevice dev, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { printf("read fail %s\n", path); exit(1); }
    fclose(f);
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = buf.size();
    ci.pCode = (const uint32_t*)buf.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

static void init_vulkan(Ctx& c, bool verbose) {
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.inst));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
    c.pd = VK_NULL_HANDLE;
    for (auto pd : pds) {
        VkPhysicalDeviceProperties prop;
        vkGetPhysicalDeviceProperties(pd, &prop);
        if (strstr(prop.deviceName, "NVIDIA")) { c.pd = pd; break; }
    }
    if (!c.pd) { printf("no NVIDIA device\n"); exit(1); }

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &extCount, exts.data());
    bool hasCoop = false;
    for (auto& e : exts) if (!strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) hasCoop = true;
    if (!hasCoop) { printf("VK_KHR_cooperative_matrix not supported\n"); exit(1); }

    auto fnProps = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(c.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fnProps) { printf("props fn not found\n"); exit(1); }
    uint32_t pc = 0;
    fnProps(c.pd, &pc, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> props(pc, { VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR });
    fnProps(c.pd, &pc, props.data());
    for (auto& p : props) {
        if (verbose) printf("coopmat prop: M=%u N=%u K=%u A=%d B=%d C=%d scope=%u\n",
               p.MSize, p.NSize, p.KSize, (int)p.AType, (int)p.BType, (int)p.CType, (uint32_t)p.scope);
        if (p.AType == VK_COMPONENT_TYPE_SINT8_KHR && p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.CType == VK_COMPONENT_TYPE_SINT32_KHR && p.MSize == 16 && p.NSize == 8 && p.KSize == 32)
            c.scope = (int)p.scope;
    }
    if (c.scope < 0) { printf("no 16x8x32 s8/s8/s32 coopmat support\n"); exit(1); }
    printf("selected scope = %d (%s)\n", c.scope, c.scope == 3 ? "subgroup" : "workgroup");

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qn, qfp.data());
    c.qf = 0xFFFFFFFF;
    for (uint32_t i = 0; i < qn; ++i)
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
    if (c.qf == 0xFFFFFFFF) { printf("no compute queue\n"); exit(1); }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = c.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE;
    f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    fcm.cooperativeMatrix = VK_TRUE;
    f12.pNext = &fcm;
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f12;

    const char* devExts[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;
    VK_CHECK(vkCreateDevice(c.pd, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);

    VkPhysicalDeviceProperties pdp;
    vkGetPhysicalDeviceProperties(c.pd, &pdp);
    c.tsPeriod = (float)pdp.limits.timestampPeriod;

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = c.qf;
    VK_CHECK(vkCreateCommandPool(c.dev, &pci, nullptr, &c.pool));
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = c.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, &c.cmd));

    VkQueryPoolCreateInfo qpi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 256;
    VK_CHECK(vkCreateQueryPool(c.dev, &qpi, nullptr, &c.qp));
}

static void create_buffers(Ctx& c, const ModelData& m, const std::vector<int8_t>& a0host) {
    Offsets& o = c.off;
    VkDeviceSize A = 256;
    auto al = [&](VkDeviceSize x) { return (x + A - 1) / A * A; };
    VkDeviceSize cur = 0;
    o.a0 = cur; cur += al((VkDeviceSize)HW * 16);
    o.x0 = cur; cur += al((VkDeviceSize)HW * 16);
    o.x1 = cur; cur += al((VkDeviceSize)HW * 32);
    o.outF = cur; cur += al((VkDeviceSize)HW * 16);
    for (int l = 0; l < 5; ++l) {
        o.w[l] = cur; cur += al((VkDeviceSize)K_GEMM[l] * C_OUT_PAD[l]);
        o.bq[l] = cur; cur += al((VkDeviceSize)C_OUT_PAD[l] * 4);
        o.mult[l] = cur; cur += al((VkDeviceSize)C_OUT_PAD[l] * 4);
    }
    o.total = cur;

    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = o.total;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(c.dev, &bci, nullptr, &c.bigbuf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c.dev, c.bigbuf, &mr);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(c.pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &c.devmem));
    VK_CHECK(vkBindBufferMemory(c.dev, c.bigbuf, c.devmem, 0));

    bci.size = o.total;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(c.dev, &bci, nullptr, &c.stgbuf));
    vkGetBufferMemoryRequirements(c.dev, c.stgbuf, &mr);
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(c.pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &c.stgmem));
    VK_CHECK(vkBindBufferMemory(c.dev, c.stgbuf, c.stgmem, 0));
    VK_CHECK(vkMapMemory(c.dev, c.stgmem, 0, VK_WHOLE_SIZE, 0, &c.stgmapped));

    memset(c.stgmapped, 0, (size_t)o.total);
    memcpy((char*)c.stgmapped + o.a0, a0host.data(), a0host.size());
    for (int l = 0; l < 5; ++l) {
        memcpy((char*)c.stgmapped + o.w[l], m.layer[l].b.data(), m.layer[l].b.size());
        memcpy((char*)c.stgmapped + o.bq[l], m.layer[l].bias_q.data(), m.layer[l].bias_q.size() * 4);
        memcpy((char*)c.stgmapped + o.mult[l], m.layer[l].mult.data(), m.layer[l].mult.size() * 4);
    }

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cmd, &bi));
    VkBufferCopy cp{ 0, 0, o.total };
    vkCmdCopyBuffer(c.cmd, c.stgbuf, c.bigbuf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(c.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(c.dev, fence, nullptr);
    VK_CHECK(vkResetCommandBuffer(c.cmd, 0));
}

static void create_pipeline_objects(Ctx& c) {
    VkDescriptorSetLayoutBinding b[NBIND] = {};
    for (int i = 0; i < NBIND; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = NBIND;
    dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &dli, nullptr, &c.dsl));

    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.dsl;
    VK_CHECK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &c.pl));

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 };
    VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 4;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool));
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = c.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &c.dsl;
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &dai, &c.ds));

    VkShaderModule mod = load_spv(c.dev, c.scope == 3 ? "build/conv_subgroup.spv" : "build/conv_workgroup.spv");

    VkDescriptorSetLayoutBinding sb[3] = {};
    for (int i = 0; i < 3; ++i) {
        sb[i].binding = i;
        sb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sb[i].descriptorCount = 1;
        sb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sdli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    sdli.bindingCount = 3;
    sdli.pBindings = sb;
    VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &sdli, nullptr, &c.smokedsl));
    VkPipelineLayoutCreateInfo spli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    spli.setLayoutCount = 1;
    spli.pSetLayouts = &c.smokedsl;
    VK_CHECK(vkCreatePipelineLayout(c.dev, &spli, nullptr, &c.smokelayout));
    dai.pSetLayouts = &c.smokedsl;
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &dai, &c.smokedb));
    VkShaderModule smod = load_spv(c.dev, c.scope == 3 ? "build/smoke_subgroup.spv" : "build/smoke_workgroup.spv");
    VkComputePipelineCreateInfo sci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    sci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    sci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    sci.stage.module = smod;
    sci.stage.pName = "main";
    sci.layout = c.smokelayout;
    VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &sci, nullptr, &c.smokepipe));
    vkDestroyShaderModule(c.dev, smod, nullptr);

    for (int l = 0; l < 5; ++l) {
        int32_t spec[8] = {
            C_IN_PAD[l], C_OUT_PAD[l], C_OUT_REAL[l], l == 4 ? 1 : 0,
            K_GEMM[l], (K_GEMM[l] + 31) / 32 * 32, (K_GEMM[l] + 31) / 32 * 32, 192
        };
        VkSpecializationMapEntry me[8];
        for (int i = 0; i < 8; ++i) { me[i].constantID = i; me[i].offset = (uint32_t)(i * 4); me[i].size = 4; }
        VkSpecializationInfo spi{ 8, me, sizeof(spec), spec };
        VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = &spi;
        ci.layout = c.pl;
        VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &c.pipes[l]));
    }
    vkDestroyShaderModule(c.dev, mod, nullptr);
}

static void set_layer_descriptors(Ctx& c, int l, const int32_t* specDummy) {
    const Offsets& o = c.off;
    VkDeviceSize inOff[5] = { o.a0, o.x0, o.x1, o.x0, o.x1 };
    VkDeviceSize out8[5] = { o.x0, o.x1, o.x0, o.x1, o.x1 };
    VkDeviceSize szIn[5] = { (VkDeviceSize)HW * 16, (VkDeviceSize)HW * 16, (VkDeviceSize)HW * 32,
                             (VkDeviceSize)HW * 16, (VkDeviceSize)HW * 16 };
    VkDeviceSize szOut8 = (VkDeviceSize)HW * C_OUT_REAL[l];

    VkDescriptorBufferInfo bi[NBIND] = {};
    for (int i = 0; i < NBIND; ++i) { bi[i].buffer = c.bigbuf; bi[i].offset = 0; bi[i].range = VK_WHOLE_SIZE; }
    bi[0].offset = inOff[l];                bi[0].range = szIn[l];
    bi[1].offset = out8[l];                 bi[1].range = szOut8;
    bi[2].offset = o.w[l];                  bi[2].range = (VkDeviceSize)K_GEMM[l] * C_OUT_PAD[l];
    bi[3].offset = o.bq[l];                 bi[3].range = (VkDeviceSize)C_OUT_PAD[l] * 4;
    bi[4].offset = o.mult[l];               bi[4].range = (VkDeviceSize)C_OUT_PAD[l] * 4;
    bi[5].offset = o.outF;                  bi[5].range = (VkDeviceSize)HW * 16;

    VkWriteDescriptorSet wr[NBIND] = {};
    for (int i = 0; i < NBIND; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = c.ds;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, NBIND, wr, 0, nullptr);
}

static void run_layer(Ctx& c, int l, bool timing, uint32_t queryBase) {
    set_layer_descriptors(c, l, nullptr);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cmd, &bi));
    if (timing) {
        vkCmdResetQueryPool(c.cmd, c.qp, queryBase, 2);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.qp, queryBase);
    }
    vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipes[l]);
    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &c.ds, 0, nullptr);
    vkCmdDispatch(c.cmd, (HW + 191) / 192, 1, 1);
    if (timing)
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.qp, queryBase + 1);
    VK_CHECK(vkEndCommandBuffer(c.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(c.dev, fence, nullptr);
    VK_CHECK(vkResetCommandBuffer(c.cmd, 0));
}

static bool download_and_cmp(Ctx& c, VkDeviceSize off, size_t bytes, const void* ref, const char* what) {
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cmd, &bi));
    VkBufferCopy cp{ off, off, (VkDeviceSize)bytes };
    vkCmdCopyBuffer(c.cmd, c.bigbuf, c.stgbuf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(c.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(c.dev, fence, nullptr);
    VK_CHECK(vkResetCommandBuffer(c.cmd, 0));
    if (memcmp((const char*)c.stgmapped + off, ref, bytes) != 0) {
        const int8_t* g = (const int8_t*)c.stgmapped + off;
        const int8_t* r = (const int8_t*)ref;
        size_t bad = 0;
        while (bad + 1 < bytes && g[bad] == r[bad]) ++bad;
        printf("[FAIL] %s first mismatch at byte %zu: got %d want %d\n",
               what, bad, (int)g[bad], (int)r[bad]);
        return false;
    }
    return true;
}

static bool smoke_test(Ctx& c) {
    static int8_t ha[32 * 32], hb[8 * 32];
    static int32_t href[16 * 8];
    for (int i = 0; i < 32 * 32; ++i) ha[i] = (int8_t)((i * 7 + 3) % 251 - 125);
    for (int i = 0; i < 8 * 32; ++i) hb[i] = (int8_t)((i * 11 + 5) % 241 - 120);
        for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            int acc = 0;
            for (int k = 0; k < 32; ++k)             acc += (int)ha[(m + 4) * 32 + k] * (int)hb[n * 32 + k];
            href[m * 8 + n] = acc;
        }

    const VkDeviceSize oa = 0, ob = 2048, oo = 3072;
    memcpy((char*)c.stgmapped + oa, ha, sizeof(ha));
    memcpy((char*)c.stgmapped + ob, hb, sizeof(hb));
    memset((char*)c.stgmapped + oo, 0, 128 * 4);

    VkDescriptorBufferInfo bi[3] = {};
    for (int i = 0; i < 3; ++i) { bi[i].buffer = c.stgbuf; bi[i].offset = 0; bi[i].range = VK_WHOLE_SIZE; }
    bi[0].offset = oa; bi[1].offset = ob; bi[2].offset = oo;
    VkWriteDescriptorSet wr[3] = {};
    for (int i = 0; i < 3; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = c.smokedb;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, 3, wr, 0, nullptr);

    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    VkCommandBufferBeginInfo binfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cmd, &binfo));
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.smokepipe);
    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.smokelayout, 0, 1, &c.smokedb, 0, nullptr);
    vkCmdDispatch(c.cmd, 1, 1, 1);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(c.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(c.dev, fence, nullptr);
    VK_CHECK(vkResetCommandBuffer(c.cmd, 0));

    const int32_t* got = (const int32_t*)((char*)c.stgmapped + oo);
    for (int i = 0; i < 128; ++i)
        if (got[i] != href[i]) {
            printf("[FAIL] smoke: mismatch at %d: got %d want %d\n", i, got[i], href[i]);
            return false;
        }
    printf("[ OK ] smoke: coopmat 16x32x8 s8 bit-exact\n");
    return true;
}

static void validate_chain(Ctx& c) {
    ModelData m;
    if (!load_model("weights.bin", "input.bin", m)) { printf("no model\n"); exit(1); }
    std::vector<int8_t> inter[4];
    std::vector<float> ref_out;
    cpu_int8(m, inter, ref_out);
    bool ok = true;
    for (int l = 0; l < 5 && ok; ++l) {
        for (int t = 0; t <= l; ++t) run_layer(c, t, false, 0);
        char name[32];
        if (l < 4) {
            snprintf(name, sizeof(name), "coopmat layer %d", l);
            size_t bytes = (size_t)HW * C_OUT_PAD[l];
            const VkDeviceSize outOff[5] = { c.off.x0, c.off.x1, c.off.x0, c.off.x1, c.off.outF };
            ok = download_and_cmp(c, outOff[l], bytes, inter[l].data(), name);
        } else {
            snprintf(name, sizeof(name), "coopmat layer 4");
            ok = download_and_cmp(c, c.off.outF, (size_t)HW * 16, ref_out.data(), name);
        }
        if (ok && l == 4) {
            double maxabs = 0;
            std::vector<float> ref_f32;
            cpu_fp32(m, ref_f32);
            const float* g = (const float*)((char*)c.stgmapped + c.off.outF);
            for (size_t i = 0; i < ref_f32.size(); ++i) {
                double d = fabs((double)g[i] - (double)ref_f32[i]);
                if (d > maxabs) maxabs = d;
            }
            printf("[ OK ] coopmat: int8 chain bit-exact (vs fp32 max_abs=%.5f)\n", maxabs);
        }
    }
}

static void bench(Ctx& c) {
    for (int i = 0; i < 3; ++i)
        for (int l = 0; l < 5; ++l) run_layer(c, l, false, 0);
    const int iters = 20;
    float acc[5] = { 0, 0, 0, 0, 0 };
    std::vector<uint64_t> ts(2);
    for (int i = 0; i < iters; ++i)
        for (int l = 0; l < 5; ++l) {
            uint32_t qb = (uint32_t)(l * 2);
            run_layer(c, l, true, qb);
            VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, qb, 2, 16, ts.data(), 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            acc[l] += (float)((double)(ts[1] - ts[0]) * c.tsPeriod / 1e6);
        }
    double bytes = (double)HW * (16 + 16) + (double)HW * (16 + 32) + (double)HW * (32 + 16) +
                   (double)HW * (16 + 16) + (double)HW * (16 + 16);
    double total = 0;
    printf("%-8s", "coopmat");
    for (int l = 0; l < 5; ++l) {
        double v = acc[l] / iters;
        total += v;
        printf("  L%d=%7.3fms", l, v);
    }
    printf("  total=%7.3fms  %.1f GB/s\n", total, bytes / (total * 1e6));
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string mode = argc > 1 ? argv[1] : "test";
    ModelData m;
    if (!load_model("weights.bin", "input.bin", m)) {
        printf("generating model (shared with conv5)...\n");
        gen_model("weights.bin", "input.bin");
        load_model("weights.bin", "input.bin", m);
    }

    Ctx c;
    init_vulkan(c, mode == "test");
    std::vector<int8_t> a0 = quant_input_nhwc16(m);
    create_buffers(c, m, a0);
    create_pipeline_objects(c);

    if (mode == "test" || mode == "all") {
        if (!smoke_test(c)) return 1;
        validate_chain(c);
    }
    if (mode == "bench" || mode == "all") bench(c);
    return 0;
}










