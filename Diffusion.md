# Diffusion

- [Diffusion](#diffusion)
  - [生成模型对比](#生成模型对比)
    - [GAN](#gan)
    - [VAE](#vae)
    - [Flow Model](#flow-model)
    - [Diffusion Model](#diffusion-model)
  - [扩散模型](#扩散模型)
    - [Forward](#forward)
    - [Reverse](#reverse)
    - [Train](#train)
    - [Sample](#sample)
  - [Conditional Diffusion Model](#conditional-diffusion-model)
    - [Classifier-Guidance](#classifier-guidance)
    - [Classifier-Free](#classifier-free)
  - [CLIP](#clip)
  - [DALL·E 2](#dalle-2)
  - [Reference](#reference)

常见的生成模型包含[GAN](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)、[VAE](https://arxiv.org/abs/1312.6114)、[Flow Models](https://proceedings.mlr.press/v97/ho19a.html)等等, 所谓生成模型, 就是给一组随机噪声, 通过某种概率模型下的变换, 输出一些具有一定语义信息的数据(比如图像、文本等)。Diffusion Model也是一种生成模型, 2020年[DDPM](https://hojonathanho.github.io/diffusion/)的发表使得图像生成领域的很多工作都开始转向Diffusion Model。

## 生成模型对比

<div align=center>
<img src="./figs/different_generation_models.jpg" width=60%/>
</div>
</br>

### GAN

**生成对抗网络**是一种基于深度学习的生成模型，能够生成新内容。GAN采用监督学习方法，使用两个子模型: 从问题域生成新数据的**生成器模型**和将数据分类为真实的(来自领域)或假的(生成的)的**鉴别器模型**。这两个模型作为竞争对手进行训练。生成器直接产生样本数据，它的对手鉴别器则试图区分从训练数据中提取的样本和从生成器中提取的样本。这个竞争过程在训练中持续进行，直到鉴别器模型有一半以上的时间无法判断真假，这意味着生成器模型正在生成非常逼真的数据。

但是这里每个模型都可以压倒另一个: 如果鉴别器太好，它将返回非常接近0或1的值，生成器则难以获得更新的梯度; 如果生成器太好，它就会利用鉴别器的弱点导致漏报。所以这两个神经网络必须具有通过各自的学习速率达到的相似的“技能水平”，这也是我们常说的GAN难以训练的原因之一。

### VAE

**变分自编码器**是一种生成模型，它“提供潜在空间中观察结果的概率描述”。简单地说，这意味着VAE将潜在属性存储为概率分布。标准的自动编码器包括2个相似的网络，一个**编码器**和一个**解码器**。编码器接受输入并将其转换为更小的表示形式，解码器可以使用该表示形式将其转换回原始输入。变分自编码器具有连续的潜在空间，这样可以使随机采样和插值更加方便。为了实现这一点，编码器的隐藏节点不输出编码向量，而是输出两个大小相同的向量: 一个**均值向量**和一个**标准差向量**。每一个隐藏的节点都认为自己是高斯分布的。我们从编码器的输出向量中采样送入解码器, 这个过程就是随机生成。这意味着即使对于相同的输入，当平均值和标准差保持不变时，实际的编码在每一次传递中都会有所不同。

训练过程是最小化**重构损失**(输出与输入的相似程度)和**潜在损失**(隐藏节点与正态分布的接近程度)。潜在损失越小，可以编码的信息就越少，这样重构损失就会增加，所以在潜在损失和重建损失之间是需要进行进行权衡的。当潜在损耗较小时，生成的图像与训练的的图像会过于相似，效果较差。在重构损失小的情况下，训练时的重构图像效果较好，但生成的新图像与重构图像相差较大，所以需要找到一个好的平衡。

VAE的一个主要缺点是它们生成的输出模糊, 这是由数据分布恢复和损失函数计算的方式造成的。

### Flow Model

**基于流的生成模型**是精确的对数似然模型，它将一堆**可逆变换**应用于来自先验的样本，以便可以计算观察的精确对数似然。与前两种算法不同，该模型显式地学习数据分布，因此损失函数是负对数似然。流模型 $f$ 被构造为一个将高维随机变量 $x$ 映射到标准高斯潜变量 $z$ 的**可逆变换**, 它可以是任意的双射函数，并且可以通过叠加各个简单的可逆变换来形成。

流模型可逆但计算效率并不高，基于流的模型生成相同分辨率的图像所需时间是GAN的几倍。

### Diffusion Model

Diffusion Model的灵感来自 *non-equilibrium thermodynamics (非平衡热力学)*, 理论首先定义扩散步骤的马尔可夫链，缓慢地将随机噪声添加到数据中，然后学习逆向扩散过程以从噪声中构造所需的数据样本。与VAE或流模型不同，扩散模型是通过固定过程学习，并且隐空间具有比较高的维度。

## 扩散模型

扩散模型(Diffusion Model)用于生成与训练数据相似的数据。从根本上说，Diffusion Model的工作原理是通过连续添加**高斯噪声**来破坏训练数据，然后通过学习反转的去噪过程来恢复数据。训练后，我们可以使用 Diffusion Model将随机采样的噪声传入模型中，通过学到的去噪过程来生成数据。

更具体地说，扩散模型是一种隐变量模型(latent variable model)，使用马尔可夫链(Markov Chain)映射到隐空间(latent space)。通过马尔科夫链，在每一个时间步 $t$ 中逐渐将噪声添加到数据 $x_i$ 中。Diffusion Model分为正向的扩散过程和反向的逆扩散过程。

### Forward

所谓前向过程，即往图片上加噪声的过程。给定图片 $x_0$ , 前向过程通过 $T$ 次累计对其添加高斯噪声，得到 $x_1, x_2, \cdots, x_T$ . 前向过程每个时刻 $t$ 只与 $t-1$ 时刻有关，所以可以看做**马尔科夫过程**, 其数学形式可以写成:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T}{q(x_t|x_{t-1})} = \prod_{t=1}^{T}{\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})}$$

其中 $\beta_1, \cdots, \beta_T$ 是高斯分布方差的超参数, 一般设置为是由 $0.0001$ 到 $0.02$ 线性插值。在扩散过程中，随着 $t$ 的增大, $x_t$ 越来越接近纯噪声。当 $T$ 足够大的时候，收敛为标准高斯噪声 $\mathcal{N}(0, \mathbf{I})$ 。

能够通过 $x_0$ 和 $\beta$ 快速得到 $x_t$ 对后续diffusion model的推断有巨大作用。首先我们假设 $\alpha_t = 1 - \beta_t$ ，并且 $\overline{\alpha_t} = \prod_{i=1}^{t}{\alpha_i}$ ，展开 $x_t$ 可以得到:

$$ x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_1 = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_2) + \sqrt{1-\alpha_t}\epsilon_1 = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + (\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_2 + \sqrt{1-\alpha_t}\epsilon_1)$$ 

其中 $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \mathbf{I})$, 根据正态分布的性质, 即 $\mathcal{N}(0, \sigma_1^2\mathbf{I}) + \mathcal{N}(0, \sigma_2^2\mathbf{I}) \sim \mathcal{N}(0, (\sigma_1^2+\sigma_2^2)\mathbf{I})$ 可以得到:

$$ x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\overline{\epsilon_2} \qquad (\overline{\epsilon_2} \sim \mathcal{N}(0, \mathbf{I}))$$ 

依次展开, 可以得到:

$$ x_t = \sqrt{\overline{\alpha_t}}\ x_0 + \sqrt{1-\overline{\alpha_t}}\ \overline{\epsilon_t} \qquad (\overline{\epsilon_t} \sim \mathcal{N}(0, \mathbf{I}))$$ 

因此，任意时刻 $x_t$ 满足 $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}}x_0, (1-\overline{\alpha_t})\mathbf{I})$.

### Reverse

如果说前向过程(forward)是**加噪**的过程，那么逆向过程(reverse)就是diffusion的**去噪**推断过程。如果我们能够逐步得到逆转后的分布 $q(x_{t-1}|x_t)$，就可以从完全的标准高斯分布 $\mathcal{N}(0, \mathbf{I})$ 还原出原图分布 $x_0$. 

<div align=center>
<img src="./figs/diffusion.jpg" width=80%/>
</div>
</br>

但实际上 $q(x_{t-1}|x_t)$ 难以显示地求解，因此我们可以利用神经网络来学习这一分布 $p_\theta(x_{t-1}|x_t)$ , 其中 $\theta$ 是神经网络的超参。

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)\mathbf{I})$$

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=T}^{1}{p_\theta(x_{t-1}|x_t)} = p(x_T)\prod_{t=T}^{1}{\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)\mathbf{I})}$$

训练过程就是学习上面公式中的 $\mu_\theta(x_t, t)$ 和 $\sigma_\theta(x_t, t)$ . 虽然我们无法得到逆转分布 $q(x_{t-1}|x_t)$ , 但是在训练过程中给定 $x_0$ , 我们可以利用贝叶斯公式求解 $q(x_{t-1}|x_t, x_0)$.

$$q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0)\frac{q(x_{t-1}|x_0)}{q({x_t|x_0})} = q(x_t|x_{t-1})\frac{q(x_{t-1}|x_0)}{q({x_t|x_0})}$$

这样就将后验概率转化为了已知的先验概率，代入前面推导的公式:

$$q(x_t|x_{t-1}) \propto \exp{\left(-\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{2(1-\alpha_t)}\right)}$$

$$q(x_{t-1}|x_0) \propto \exp{\left(-\frac{(x_{t-1}-\sqrt{\overline{\alpha_{t-1}}}x_0)^2}{2(1-\overline{\alpha_{t-1}})}\right)}$$

$$q(x_t|x_0) \propto \exp{\left(-\frac{(x_t-\sqrt{\overline{\alpha_t}}x_0)^2}{2(1-\overline{\alpha_t})}\right)}$$

整理可以得到:

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}_{t-1}(x_t), \tilde{\beta}_{t-1}\mathbf{I})$$

其中:

$$\tilde{\mu}_{t-1}(x_t) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_t}}x_t + \frac{\sqrt{\overline{\alpha_{t-1}}}\beta_t}{1-\overline{\alpha_t}}x_0 = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\overline{\epsilon_t})$$

$$\tilde{\beta}_{t-1} = \frac{1-\overline{\alpha_{t-1}}}{1-\overline{\alpha_t}}\beta_t \approx \beta_t$$

以上推导的 $\tilde\mu_{t-1}(x_t)$ 可视为`ground truth`, 而我们将通过神经网络学习到 $\mu_\theta(x_t, t)$ , 本质上也就是学习噪声 $\epsilon_\theta(x_t, t)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t))$$

因此模型预测的 $x_{t-1}$ 可以写成:

$$x_{t-1}(x_{t}, t; \theta) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t)) + \sigma_\theta(x_t, t)z \qquad z \sim \mathcal{N}(0, \mathbf{I})$$

### Train

训练过程就是学习上面公式中的 $\mu_\theta(x_t, t)$ 和 $\sigma_\theta(x_t, t)$ , 进一步也就是学习噪声 $\epsilon_\theta(x_t, t)$. Diffusion使用极大似然估计来找到逆扩散过程中马尔科夫链转换的概率分布。

$$\mathcal{L} = \mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)]$$

求模型的极大似然估计，等同于求解最小化负对数似然的变分上限 $\mathcal{L}_{vlb}$:

$$\mathcal{L} = \mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)] \leq \mathbb{E}_{q(x_{0\ :\ T})}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right] := \mathcal{L}_{vlb}$$

进一步表示为KL散度(`KL散度是一种不对称统计距离度量，用于衡量一个概率分布P与另外一个概率分布Q的差异程度`):

$$\mathcal{L}_{vlb} = \mathbb{E}_{q(x_{0\ :\ T})}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right] = \mathbb{E}_{q(x_{0\ :\ T})}\left[\log \left(\prod_{t=1}^{T}{q(x_t|x_{t-1})}\right) / \left(p_\theta(x_T)\prod_{t=1}^{T}{p_\theta(x_{t-1}|x_t)}\right)\right]$$

$$= \mathbb{E}_{q(x_{0\ :\ T})}\left[-\log p_\theta(x_T) + \sum_{t=1}^{T}{\log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}}\right] \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$

$$ = \mathbb{E}_{q(x_{0\ :\ T})}\left[-\log p_\theta(x_T) + \sum_{t=2}^{T}{\log \left(\frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} \right)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)} \right] \quad$$

$$\quad = \mathbb{E}_{q(x_{0\ :\ T})}\left[-\log p_\theta(x_T) + \sum_{t=2}^{T}{\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}} +\sum_{t=2}^{T}{\log \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)} \right]$$

$$ = \mathbb{E}_{q(x_{0\ :\ T})}\left[-\log p_\theta(x_T) + \sum_{t=2}^{T}{\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}} + \log \frac{q(x_T|x_0)}{q(x_1|x_0)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)} \right]\quad$$

$$= \mathbb{E}_{q(x_{0\ :\ T})}\left[\log \frac{q(x_T|x_0)}{p_\theta(x_T)} + \sum_{t=2}^{T}{\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}} - \log p_\theta(x_0|x_1)\right] \qquad\qquad\qquad\qquad$$

由于前向 $q$ 没有可学习参数，而 $x_T$ 则是纯高斯噪声, 因此上式第一项为一常量，可以忽略; 第三项是由连续变为离散的熵，对于一般的连续情况可以合并进第二项, 因此:

$$\mathcal{L}_{vlb} = \mathbb{E}_{q(x_{0\ :\ T})}\left[\sum_{t=1}^{T}{\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}}\right] + C := \sum_{t=1}^{T}{\mathcal{L}_t}+C$$

$\mathcal{L}_t$ 是两个高斯分布的KL散度。根据[多元高斯分布的KL散度求解公式](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Kullback%25E2%2580%2593Leibler_divergence%23Multivariate_normal_distributions):

$$\mathcal{L}_t = \mathbb{E}_{q(x_{0\ :\ T})}\left[ \frac{||\tilde{\mu_t}(x_t)-\mu_\theta(x_t, t)||^2}{2||\sigma_\theta^2(x_t, t)\mathbf{I}||^2} \right] + C^\prime$$

代入上面推导的 $\tilde{\mu_t}$ 和 $\mu_\theta$ 计算公式，进一步化简得到:

$$\mathcal{L}_t^{simple} = \mathbb{E}_{x_0, t, \epsilon}\left[||\epsilon-\epsilon_\theta(\sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}\epsilon, t)||^2 \right]$$

训练的核心就是最小化模型预测噪声 $\epsilon_\theta$ 与实际噪声 $\epsilon$. 训练过程的伪代码如下:

```python
x0 = get_data()
epsilon = torch.randn_like(x0.shape)
t = torch.randint()

x_t = x0 * torch.sqrt(alpha_bar[t]) + epsilon * torch.sqrt(1 - alpha_bar[t])

output = model(x_t, t)

loss = torch.norm(epsilon - output)**2

loss.backward()
```

### Sample

采样过程就是所谓的推断过程，给定一个噪声图片 $x_T$ ，通过公式:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)\mathbf{I}) \sim \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t)) + \sigma_\theta(x_t, t)z \qquad (z \sim \mathcal{N}(0, \mathbf{I}))$$

从 $t=T$ 开始逐步去噪，直至生成图像 $x_0$ . 在实际应用中，我们可以近似 $\beta_t \approx \sigma_\theta^2(x_t, t)$ ，算法伪代码如下:

```python
x_T = torch,randn_like(x0.shape)

for t in range(T, 0, -1):
    e = model(x_T, t)
    mu = 1/torch.sqrt(alpha_bar[t]) * (x_T - beta[t]/torch.sqrt(1-alpha_bar[t]) * e)
    sigma = torch.sqrt(beta[t])
    x_T = mu + sigma * torch.randn_like(x0.shape)

return x_T
```

## Conditional Diffusion Model

作为生成模型，扩散模型跟VAE、GAN、flow等模型的发展史很相似，都是先出来了无条件生成，然后有条件生成就紧接而来。无条件生成往往是为了探索效果上限，而有条件生成则更多是应用层面的内容，因为它可以实现根据我们的意愿来控制输出结果。

从方法上来看，条件控制生成的方式分两种：**事后修改(Classifier-Guidance)**和**事前训练(Classifier-Free)**。对于大多数人来说，一个SOTA级别的扩散模型训练成本太大了，而分类器（Classifier）的训练还能接受，所以就想着直接复用别人训练好的无条件扩散模型，用一个分类器来调整生成过程以实现控制生成，这就是事后修改的Classifier-Guidance方案；而对于“财大气粗”的Google、OpenAI等公司来说，它们不缺数据和算力，所以更倾向于往扩散模型的训练过程中就加入条件信号，达到更好的生成效果，这就是事前训练的Classifier-Free方案。

Classifier-Guidance方案最早出自[《Diffusion Models Beat GANs on Image Synthesis》](https://arxiv.org/abs/2105.05233)，最初就是用来实现按类生成; 后来[《More Control for Free! Image Synthesis with Semantic Diffusion Guidance》](https://arxiv.org/abs/2112.05744)推广了 “Classifier” 的概念，使得它也可以按图、按文来生成。Classifier-Guidance方案的训练成本比较低，但是推断成本会高些，而且控制细节上通常没那么到位。

至于Classifier-Free方案，最早出自[《Classifier-Free Diffusion Guidance》](https://arxiv.org/abs/2207.12598)，后来的[Imagen](https://arxiv.org/abs/2205.11487)等吸引人眼球的模型基本上都是以它为基础做的。应该说，Classifier-Free方案本身没什么理论上的技巧，它是条件扩散模型最朴素的方案，出现得晚只是因为重新训练扩散模型的成本较大吧。在数据和算力都比较充裕的前提下，Classifier-Free方案表现出了令人惊叹的细节控制能力。

### Classifier-Guidance

这一方案是在已经训练好的diffusion model上额外添加一个分类器 $p_\phi(y|x_t)$ 用于引导, 其中 $y$ 是条件, $\phi$ 是分类器的参数。这一分类器再被应用前需要先用带噪声的图片 $x_t$ 训练, 具体来说, 可以通过diffusion model对原始图片进行加噪处理(正向传播), 把得到的噪声图片喂给classifier做训练。下面我们主要考虑如何使用这个分类器来引导diffusion生成。

无条件生成过程可以表述为 $p_\theta(x_{t-1}|x_t)$ , 加上条件 $y$ 后可以被写成 $p_{\theta,\phi}(x_{t-1}|x_t, y)$ , 利用贝叶斯公式:

$$p_{\theta,\phi}(x_{t-1}|x_t, y) = \frac{p_\theta(x_{t-1}|x_t)p_\phi(y|x_{t-1},x_t)}{p_\phi(y|x_t)}$$

由于 $x_t$ 只是在 $x_{t-1}$ 基础上添加噪声, 对分类不会有影响, 因此有 $p_\phi(y|x_{t-1},x_t) = p_\phi(y|x_{t-1})$ , 代入得:

$$p_{\theta,\phi}(x_{t-1}|x_t, y) = p_\theta(x_{t-1}|x_t)\frac{p_\phi(y|x_{t-1})}{p_\phi(y|x_t)} = p_\theta(x_{t-1}|x_t)e^{\log p_\phi(y|x_{t-1}) - \log p_\phi(y|x_t)}$$

当 $T$ 足够大时, $x_{t-1}$ 和 $x_t$ 相差很小, 因此 $p_\phi(y|x_{t-1})$ 和 $p_\phi(y|x_t)$ 也相差很小, 故可以对其做泰勒展开:

$$\log p_\phi(y|x_{t-1}) - \log p_\phi(y|x_t) \approx (x_{t-1} - x_t) \nabla_{x_t} \log p_\phi(y|x_t) \approx (x_{t-1} - \mu_\theta(x_t,t)) \nabla_{x_t} \log p_\phi(y|x_t)$$

由于 $p_\theta(x_{t-1}|x_t) \sim \mathcal{N}(x_t; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)\mathbf{I}) \propto \exp \left( - \frac{(x_t - \mu_\theta(x_t, t))^2}{2\sigma_\theta^2(x_t, t)} \right)$ , 因此有:

$$p_{\theta,\phi}(x_{t-1}|x_t, y) \propto \exp \left( - \frac{(x_t - \mu_\theta(x_t, t))^2}{2\sigma_\theta^2(x_t, t)} + (x_{t-1} - \mu_\theta(x_t,t)) \nabla_{x_t} \log p_\phi(y|x_t) \right) \propto \exp \left( - \frac{(x_t - \mu_\theta(x_t, t) - \sigma_\theta^2(x_t, t)\nabla_{x_t} \log p_\phi(y|x_t) )^2}{2\sigma_\theta^2(x_t, t)}\right)$$

也即: 

$$p_{\theta,\phi}(x_{t-1}|x_t, y) = \mathcal{N}(x_t; \mu_\theta(x_t, t) + \sigma_\theta^2(x_t, t)\nabla_{x_t} \log p_\phi(y|x_t), \sigma_\theta^2(x_t, t)\mathbf{I}) \sim \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t)) + \sigma_\theta^2(x_t, t)\nabla_{x_t} \log p_\phi(y|x_t) + \sigma_\theta(x_t, t) z \qquad (z \sim \mathcal{N}(0, \mathbf{I}))$$

与不加条件的情形相比仅仅在均值里多了一项 $\sigma_\theta^2(x_t, t)\nabla_{x_t} \log p_\phi(y|x_t)$ . 在实际sample过程中，用classifier对diffusion model生成的图片 $x_t$ 进行分类，得到预测分数与目标类别的交叉熵 $\log p_\phi(y|x_t)$ ，把它对 $x_t$ 求梯度，用梯度引导下一步的生成采样。另外，我们也可以在这一项前面添加调节因子 $\gamma$ 来控制生成图片与 $y$ 的相似性。

### Classifier-Free

这种方法需要对模型进行重新训练，与无条件diffusion model相比，输入除了高斯噪声 $x_T$ 还包含条件向量 $y$ , 相应的公式也要做出调整:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t, y))$$

优化目标为:

$$\mathbb{E}_{x_0, y\sim p(x_0, y), t, \epsilon}\left[||\epsilon-\epsilon_\theta(\sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}\epsilon, y, t)||^2 \right]$$

类似于Classifier-Guidance添加调节因子 $\gamma$ , Classifier-Free也可以采用类似的技巧:

$$ \tilde{\epsilon_\theta}(x_t, t, y) = (1+w)\epsilon_\theta(x_t, t, y) - w\epsilon_\theta(x_t, t)$$

其中 $w$ 是类似 $\gamma$ 的调节因子。 $\tilde{\epsilon_\theta}$ 包含conditional和unconditional两部分，在训练时我们以一定概率将 $y$ 置为 None 来训练unconditional的部分，在采样时用 $\tilde{\epsilon_\theta}$ 替换原来的 $\epsilon_\theta$ 计算。

## CLIP

[CLIP](https://arxiv.org/abs/2103.00020) (Constastive Language-Image Pretraining) 是一个多模态预训练的算法，它连接了图像与文本，提供一张图像和一段文本描述，该模型可以预测与该图像最相关的文本描述，而不需要为某个特定任务进行优化。该模型的架构如下：

<div align=center>
<img src="./figs/clip.png" width=90%/>
</div>
</br>

在训练时，将 $N$ 个 <文本，图像> 对作为输入，其中文本信息通过text encoder (如Transformer) 编码为 $N \times d$ 的向量，图像信息通过image encoder (如ResNet，Vision Transformer) 编码为 $N \times d$ 的向量，然后计算文本向量与图像向量的内积得到 $N \times N$ 矩阵，此时对角线上的元素表示输入的 $N$ 个 <文本，图像> 对的相似度。我们希望对角线上的元素最大化，其余元素最小化，这可以通过交叉熵进行优化。

在预测时，我们提供一组文本描述，通过text encoder得到文本向量，然后给定一张图像，通过image encoder得到图片向量，最后计算文本向量与所有文本向量的内积，选择内积最大的文本(也即最相似)作为预测结果。

CLIP是一个zero-shot的图片分类器，这意味着不用额外训练就可以分类任意类别的图片。

## DALL·E 2

[DALL·E 2](https://arxiv.org/abs/2204.06125) 是一个基于CLIP和diffusion的文本到图片的生成模型。它的架构如下：

<div align=center>
<img src="./figs/dalle2.png" width=90%/>
</div>
</br>

它主要包括三个部分：CLIP，prior和decoder。DALL·E 2将三个子模块分开训练，最后将这些训练好的子模块拼接在一起。训练过程如下：
- 第一步，训练CLIP，得到一个文本编码器和一个图像编码器。训练过程与标准的CLIP训练过程一致。
- 第二步，训练prior，使得其可以根据输入到text embedding生成图片embedding。具体来说，将CLIP中训练好的text encoder拿出来，输入文本 $y$ , 得到文本编码 $z_t$ ; 将CLIP中训练好的image encoder拿出来，输入图像 $x$ , 得到图像编码 $z_i$ . 假设 $z_t$ 经过prior输出的特征为 $z_i^{\prime}$ , 那么我们自然希望 $z_i^{\prime}$ 与 $z_i$ 越接近越好，以此来更新prior模块。prior可以是Autoregressive model也可以是conditional diffusion model.
- 第三步，训练decoder, 使得其根据图像编码 $z_i$ 还原出图片 $x$ . 具体来说，将CLIP中训练好的image encoder拿出来，输入图像 $x$ , 得到图像编码 $z_i$ , 将 $z_i$ 送入到decoder中得到图片 $x^{\prime}$ , 我们希望 $x^{\prime}$ 与 $x$ 的特征尽量接近。论文中使用的decoder是conditional diffusion model.

在做推理时，给定一个文本 $y$ , 通过CLIP text encoder得到文本向量 $z_t$ , 然后通过prior得到图像向量 $z_i$ , 最后通过decoder得到图片 $x$ .

## Reference

- [Diffusion Models：生成扩散模型](https://zhuanlan.zhihu.com/p/549623622)
- [由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459)
- [生成扩散模型(一): 基础 (Generative Diffusion Model: Basic)](https://www.jarvis73.com/2022/08/08/Diffusion-Model-1/)
- [生成扩散模型漫谈（九）：条件控制生成结果](https://spaces.ac.cn/archives/9257)
- [基于扩散模型的文本引导图像生成算法](https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/124641910)
- [AIGC 神器 CLIP：技术详解及应用示例](https://xie.infoq.cn/article/f7680d1fe54f4a0e2db7acbd3)
- [DALL·E 2 解读 | 结合预训练CLIP和扩散模型实现文本-图像生成](https://blog.csdn.net/zcyzcyjava/article/details/126992705)