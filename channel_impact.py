import numpy as np
import torch
import random
'''
function used to add noise in paper 'Emergent Semantic Communications:
Basic Ideas, Challenges, and Opportunities for Tactile Internet' is 'noisy_channel(word,epsi)' and 'message_in_channel2(word,epsi)'.
message_in_channel2 is responsible for spliting the torch.tensor type data into several executable subsets.
noisy_channel is responsible for adding noise.
epsi represent epsilon, the error rate.
'''

def noisy_channel(word,epsi):
    vocab_size=len(word)
    a = random.random()
    if a <= (1 - epsi):
        new_word=word
    else:
        shift=random.randint(1,vocab_size-1)
        index = torch.where(word == 1)
        compare=index[0]+shift
        if compare>vocab_size-1:
            real_index=compare-vocab_size
        else:
            real_index=index[0]+shift
        new_word = word.clone()
        new_word[index]=0
        new_word[real_index]=1
    return new_word

# Pulse Amplitude Modulation
def channel_impact_PAM(bits):
    # 原始数字信号
    bits = np.array(bits)

    # 调制参数
    M = 4  # PAM阶数
    fc = 10  # 载波频率

    # 调制
    t = np.arange(len(bits))
    carrier = np.cos(2 * np.pi * fc * t)
    pam = (2 * bits - 1) * M  # 将0和1映射为-PAM和+PAM
    modulated = carrier * pam

    # 添加噪声
    noise_power = 0.1
    noisy_signal = modulated + np.random.normal(0, np.sqrt(noise_power), len(modulated))

    # 解调
    demodulated = noisy_signal / carrier
    decisions = (demodulated > 0).astype(int)  # 判决阈值为0，得到离散的数字信号
    return decisions

    # 绘图
    # fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    # axs[0].plot(t, bits, '-o')
    # axs[0].set_title('Original Signal')
    # axs[1].plot(t, pam, '-o')
    # axs[1].set_title('PAM Signal')
    # axs[2].plot(t, modulated, '-o')
    # axs[2].set_title('Modulated Signal')
    # axs[3].plot(t, noisy_signal, '-o')
    # axs[3].set_title('Noisy Signal')
    # plt.tight_layout()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    # ax.step(t, decisions, '-o')
    # ax.set_title('Demodulated Signal')
    # plt.tight_layout()
    #
    # plt.show()


def message_in_channel2(original_message,epsil):
    higher_tensors=[]
    for m in original_message:
        tensors=[]
        for n in m:
            current_message = n
            message_midle = noisy_channel(current_message,epsil)
            tensors.append(message_midle.unsqueeze(0))
        concatenated = torch.cat(tensors, dim=0)
        higher_tensors.append(concatenated.unsqueeze(0))
    final_message=torch.cat(higher_tensors, dim=0)
    return final_message


# torch.tensor is a data type that usually used in machine learning training.
# For a test data 'a', there are four messages in 'a', each message is composed of four characters.

# 生成一个随机的列索引，使得每一行对应的该列的值为1
test = torch.zeros(1, 11760, 4096)
random_column_indices = torch.randint(0, 4096, (11760,))
test[:, torch.arange(11760), random_column_indices] = 1

a=torch.tensor([[[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
       device='cuda:0')
print(test.shape)
aa=message_in_channel2(test,0.1)
print(aa.squeeze(0).shape)

# Code written by ---- Ruxiao Chen
