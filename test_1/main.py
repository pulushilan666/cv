import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageProcessor:
    def __init__(self, image_path):
        """初始化图像处理器"""
        # 读取图像
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式（用于显示和直方图）
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        # 转换为灰度图（用于滤波）
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # 定义Sobel算子
        self.sobel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=np.float32)

        self.sobel_y = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=np.float32)

        # 定义给定的卷积核
        self.given_kernel = np.array([[1, 0, -1],
                                      [2, 0, -2],
                                      [1, 0, -1]], dtype=np.float32)

    def convolution(self, image, kernel, padding='zero'):
        """手动实现卷积操作

        参数:
            image: 输入图像（灰度图）
            kernel: 卷积核
            padding: 填充方式，'zero'或'same'

        返回:
            卷积后的图像
        """
        # 获取图像和卷积核的尺寸
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape

        # 计算填充大小
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        # 填充图像
        if padding == 'zero':
            padded_image = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w), dtype=np.float32)
            padded_image[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image
        else:  # same padding
            padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'edge')

        # 初始化输出图像
        output = np.zeros_like(image, dtype=np.float32)

        # 执行卷积
        for i in range(img_h):
            for j in range(img_w):
                # 提取当前窗口
                window = padded_image[i:i + kernel_h, j:j + kernel_w]
                # 计算卷积结果
                output[i, j] = np.sum(window * kernel)

        # 归一化到0-255范围
        output = np.clip(output, 0, 255)

        return output.astype(np.uint8)

    def sobel_filter(self):
        """应用Sobel算子滤波"""
        # 分别计算x方向和y方向的梯度
        grad_x = self.convolution(self.gray_image, self.sobel_x)
        grad_y = self.convolution(self.gray_image, self.sobel_y)

        # 计算梯度幅值
        sobel_magnitude = np.sqrt(grad_x.astype(np.float32) ** 2 + grad_y.astype(np.float32) ** 2)
        sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)

        return sobel_magnitude, grad_x, grad_y

    def given_kernel_filter(self):
        """应用给定卷积核滤波"""
        return self.convolution(self.gray_image, self.given_kernel)

    def compute_color_histogram(self, bins=256):
        """计算颜色直方图（不调用函数包）"""
        # 分离RGB通道
        r_channel = self.rgb_image[:, :, 0]
        g_channel = self.rgb_image[:, :, 1]
        b_channel = self.rgb_image[:, :, 2]

        # 初始化直方图数组
        r_hist = np.zeros(bins, dtype=np.int32)
        g_hist = np.zeros(bins, dtype=np.int32)
        b_hist = np.zeros(bins, dtype=np.int32)

        # 计算每个通道的直方图
        height, width = r_channel.shape

        for i in range(height):
            for j in range(width):
                r_val = r_channel[i, j]
                g_val = g_channel[i, j]
                b_val = b_channel[i, j]

                r_hist[r_val] += 1
                g_hist[g_val] += 1
                b_hist[b_val] += 1

        return r_hist, g_hist, b_hist

    def extract_texture_features(self, distances=[1], angles=[0]):
        """
        提取纹理特征（基于灰度共生矩阵）

        参数:
            distances: 像素对之间的距离
            angles: 像素对之间的角度（弧度）

        返回:
            纹理特征向量
        """
        # 获取灰度图像
        gray = self.gray_image

        # 量化灰度级（减少计算量）
        levels = 8
        quantized = (gray // (256 // levels)).astype(np.uint8)

        # 初始化灰度共生矩阵
        glcm = np.zeros((levels, levels), dtype=np.float32)

        height, width = quantized.shape
        distance = distances[0]

        # 计算灰度共生矩阵（0度方向）
        for i in range(height):
            for j in range(width - distance):
                row = quantized[i, j]
                col = quantized[i, j + distance]
                glcm[row, col] += 1

        # 归一化GLCM
        glcm_sum = np.sum(glcm)
        if glcm_sum > 0:
            glcm = glcm / glcm_sum

        # 计算纹理特征
        features = {}

        # 1. 对比度 (Contrast)
        contrast = 0
        for i in range(levels):
            for j in range(levels):
                contrast += glcm[i, j] * (i - j) ** 2
        features['contrast'] = contrast

        # 2. 能量 (Energy) / 均匀性 (Homogeneity)
        energy = 0
        for i in range(levels):
            for j in range(levels):
                energy += glcm[i, j] ** 2
        features['energy'] = energy

        # 3. 同质性 (Homogeneity)
        homogeneity = 0
        for i in range(levels):
            for j in range(levels):
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        features['homogeneity'] = homogeneity

        # 4. 熵 (Entropy)
        entropy = 0
        for i in range(levels):
            for j in range(levels):
                if glcm[i, j] > 0:
                    entropy -= glcm[i, j] * np.log(glcm[i, j])
        features['entropy'] = entropy

        # 5. 相关性 (Correlation)
        # 计算均值和标准差
        i_vals, j_vals = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
        mean_i = np.sum(i_vals * glcm)
        mean_j = np.sum(j_vals * glcm)

        std_i = np.sqrt(np.sum((i_vals - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j_vals - mean_j) ** 2 * glcm))

        if std_i > 0 and std_j > 0:
            correlation = np.sum((i_vals - mean_i) * (j_vals - mean_j) * glcm) / (std_i * std_j)
        else:
            correlation = 0
        features['correlation'] = correlation

        return features, glcm

    def save_texture_features(self, features, filename='texture_features.npy'):
        """保存纹理特征到npy文件"""
        # 将特征字典转换为numpy数组
        feature_vector = np.array(list(features.values()))
        np.save(filename, feature_vector)
        print(f"纹理特征已保存到: {filename}")
        return feature_vector

    def visualize_results(self, sobel_result, given_kernel_result, r_hist, g_hist, b_hist):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 显示原始图像
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 显示Sobel滤波结果
        axes[0, 1].imshow(sobel_result, cmap='gray')
        axes[0, 1].set_title('Sobel算子滤波')
        axes[0, 1].axis('off')

        # 显示给定卷积核滤波结果
        axes[0, 2].imshow(given_kernel_result, cmap='gray')
        axes[0, 2].set_title('给定卷积核滤波')
        axes[0, 2].axis('off')

        # 显示颜色直方图
        bins = np.arange(256)
        axes[1, 0].plot(bins, r_hist, color='red', alpha=0.7, label='Red')
        axes[1, 0].plot(bins, g_hist, color='green', alpha=0.7, label='Green')
        axes[1, 0].plot(bins, b_hist, color='blue', alpha=0.7, label='Blue')
        axes[1, 0].set_title('颜色直方图')
        axes[1, 0].set_xlabel('像素值')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 显示灰度共生矩阵
        features, glcm = self.extract_texture_features()
        axes[1, 1].imshow(glcm, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('灰度共生矩阵')
        axes[1, 1].set_xlabel('灰度级')
        axes[1, 1].set_ylabel('灰度级')

        # 显示纹理特征值
        axes[1, 2].axis('off')
        feature_text = "纹理特征值:\n"
        for key, value in features.items():
            feature_text += f"{key}: {value:.4f}\n"
        axes[1, 2].text(0.1, 0.5, feature_text, fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('纹理特征')

        plt.tight_layout()
        plt.savefig('实验结果显示.png', dpi=150, bbox_inches='tight')
        plt.show()

    def process_image(self):
        """处理图像的主函数"""
        print("开始处理图像...")

        # 1. Sobel滤波
        print("1. 应用Sobel算子滤波...")
        sobel_result, grad_x, grad_y = self.sobel_filter()

        # 2. 给定卷积核滤波
        print("2. 应用给定卷积核滤波...")
        given_kernel_result = self.given_kernel_filter()

        # 3. 计算颜色直方图
        print("3. 计算颜色直方图...")
        r_hist, g_hist, b_hist = self.compute_color_histogram()

        # 4. 提取纹理特征
        print("4. 提取纹理特征...")
        texture_features, glcm = self.extract_texture_features()

        # 5. 保存纹理特征
        print("5. 保存纹理特征...")
        feature_vector = self.save_texture_features(texture_features)

        # 6. 可视化结果
        print("6. 可视化结果...")
        self.visualize_results(sobel_result, given_kernel_result, r_hist, g_hist, b_hist)

        # 7. 保存处理后的图像
        cv2.imwrite('sobel_filtered.png', sobel_result)
        cv2.imwrite('given_kernel_filtered.png', given_kernel_result)
        print(f"Sobel滤波结果已保存到: sobel_filtered.png")
        print(f"给定卷积核滤波结果已保存到: given_kernel_filtered.png")

        print("\n处理完成!")

        # 返回处理结果
        results = {
            'original_image': self.rgb_image,
            'sobel_result': sobel_result,
            'given_kernel_result': given_kernel_result,
            'color_histogram': (r_hist, g_hist, b_hist),
            'texture_features': texture_features,
            'glcm': glcm,
            'feature_vector': feature_vector
        }

        return results


def main():
    """主函数"""
    # 图像路径 - 请替换为您自己拍摄的图像路径
    image_path = 'input_image.jpg'  # 请确保图像存在

    # 如果图像不存在，创建示例图像
    if not os.path.exists(image_path):
        print(f"图像 {image_path} 不存在，创建示例图像...")
        create_sample_image()

    try:
        # 创建图像处理器
        processor = ImageProcessor(image_path)

        # 处理图像
        results = processor.process_image()

        # 打印纹理特征
        print("\n提取的纹理特征:")
        for key, value in results['texture_features'].items():
            print(f"{key}: {value:.4f}")

    except Exception as e:
        print(f"处理图像时出错: {e}")
        print("请确保图像路径正确，图像格式为常见格式（jpg, png等）")


def create_sample_image():
    """创建示例图像（如果用户没有提供自己的图像）"""
    # 创建一个包含纹理的示例图像
    height, width = 400, 600

    # 创建一个渐变色背景
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # 生成RGB通道
    r_channel = (np.sin(X) * 127 + 128).astype(np.uint8)
    g_channel = (np.cos(Y) * 127 + 128).astype(np.uint8)
    b_channel = (np.sin(X + Y) * 127 + 128).astype(np.uint8)

    # 合并通道
    sample_image = np.stack([r_channel, g_channel, b_channel], axis=2)

    # 添加一些纹理
    texture = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    sample_image = cv2.addWeighted(sample_image, 0.7, texture, 0.3, 0)

    # 保存示例图像
    cv2.imwrite('input_image.jpg', cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    print("已创建示例图像: input_image.jpg")


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 运行主程序
    main()