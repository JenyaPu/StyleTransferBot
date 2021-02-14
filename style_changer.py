from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import copy

# Настройка вычислений на видеокарте или процессоре
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    device = torch.device("cpu")
else:
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda")
# Предустановка настроек и запуск функции
unloader = transforms.ToPILImage()
content_layers_default = 'conv_4'
style_layers_default = ('conv_1', 'conv_2', 'conv_3', 'conv_5')
content_image_name = 'photo1'
style_image_name = 'photo2'
n_iter = 300
style_weight = 50000
content_weight = 1
target_resolution = ()


# Функция, которая загружает изображение в видеокарту
def load_image(name, type1='style'):
    global target_resolution
    if type1 == 'style':
        img_name = f'{name}.jpg'
    else:
        img_name = f'{name}.jpg'
    data = Image.open(img_name)
    if type1 == 'content' and target_resolution == ():
        temp = data.size
        target_resolution = (temp[1], temp[0])
    refactor = transforms.Compose([
        transforms.Resize(target_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor()])
    data = refactor(data).unsqueeze(0)
    return data.to(device, torch.float)


# Функция потерь контента
class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = f.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input1):
        self.loss = f.mse_loss(input1, self.target)
        return input1


# Функция потерь для простого случая - наложение одного стиля на изображение
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = f.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input1):
        g = gram_matrix(input1)
        self.loss = f.mse_loss(g, self.target)
        return input1


# Функция потерь для двух конкурирующих стилей
class StyleLoss2(nn.Module):
    def __init__(self, target_feature1, target_feature2):
        super(StyleLoss2, self).__init__()
        self.target1 = gram_matrix(target_feature1).detach()
        self.target2 = gram_matrix(target_feature2).detach()
        self.loss1 = f.mse_loss(self.target1, self.target1)  # to initialize with something
        self.loss2 = f.mse_loss(self.target2, self.target2)
        self.loss = self.loss1 + self.loss2

    def forward(self, input1):
        g = gram_matrix(input1)
        self.loss1 = f.mse_loss(g, self.target1)
        self.loss2 = f.mse_loss(g, self.target2)
        self.loss = self.loss1 + self.loss2
        return input1


# Функция потерь для соседствующих на разных частях изображения стилей (применение маски)
class StyleLoss3(nn.Module):
    def __init__(self, target_feature1, target_feature2):
        super(StyleLoss3, self).__init__()
        self.target1 = gram_matrix(target_feature1).detach()
        self.target2 = gram_matrix(target_feature2).detach()
        self.loss1 = f.mse_loss(self.target1, self.target1)
        self.loss2 = f.mse_loss(self.target2, self.target2)
        self.loss = self.loss1 + self.loss2

    def forward(self, input1):
        x = torch.zeros_like(input1)
        x[0][:][:input1.size(2) // 2] = 1.
        # new_mask = torch.stack([x for _ in range(input1.shape[1])], dim=0).unsqueeze(0)
        # new_mask = new_mask.resize_(input1.shape)
        g = gram_matrix(input1 * x)
        self.loss1 = f.mse_loss(g, self.target1)
        self.loss2 = f.mse_loss(g, self.target2)
        self.loss = self.loss1 + self.loss2
        return input1


# Нормализация изображения
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img1):
        return (img1 - self.mean) / self.std


# Грам-матрица
def gram_matrix(input1):
    batch_size, f_map_num, h, w = input1.size()
    features = input1.view(batch_size * h, w * f_map_num)
    g = torch.mm(features, features.t())
    return g.div(batch_size * h * w * f_map_num)


# Построение модели для ситуации с одним стилем
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    j = 0  # проход по сверточным слоям
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            j += 1
            name = 'conv_{}'.format(j)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(j)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(j)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(j)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layers:
            # добавить content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(j), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # добавить style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(j), style_loss)
            style_losses.append(style_loss)

    # выбрасываем все уровни после последенего style loss или content loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:(j + 1)]
    return model, style_losses, content_losses


# Построение модели для ситуации с двумя конкурирующими стилями
def get_style_model_and_losses2(cnn, normalization_mean, normalization_std,
                                style_img1, style_img2, content_img,
                                content_layers,
                                style_layers):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    j = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            j += 1
            name = 'conv_{}'.format(j)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(j)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(j)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(j)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(j), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature1 = model(style_img1).detach()
            target_feature2 = model(style_img2).detach()
            # Здесь происходит выбор loss-функции
            style_loss = StyleLoss3(target_feature1, target_feature2)
            model.add_module("style_loss_{}".format(j), style_loss)
            style_losses.append(style_loss)
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss3):
            break
    model = model[:(j + 1)]
    return model, style_losses, content_losses


# содержимое тензора в список изменяемых оптимизатором параметров
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# главный цикл обучения модели
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img1, input_img,
                       num_steps=150, style_weight1=50000, content_weight1=1):
    print('Building the style transfer model..')

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, content_img, style_img1,
        content_layers_default, style_layers_default)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = torch.empty(1, 1).to(device)
            # style_score = 0
            content_score = torch.empty(1, 1).to(device)
            # content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight1
            content_score *= content_weight1
            loss = style_score + content_score
            loss.backward(retain_graph=True)
            run[0] += 1
            if run[0] % 50 == 0:
                print("Iteration {}:   ".format(run), end='')
                print('Style Loss : {:4f}; Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img


# Фунция для отрисовки всех рисунков на одном поле
def make_global_image(output, content_image_name1, style_image_name11, style_image_name21):
    image = output.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    fig = plt.figure(constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.95, wspace=1.2)
    ax4 = fig.add_subplot(gs1[:-1, :])
    ax1 = fig.add_subplot(gs1[-1, 0])
    ax3 = fig.add_subplot(gs1[-1, 1])
    ax2 = fig.add_subplot(gs1[-1, 2])
    ax1.imshow(img.imread("images/" + style_image_name11 + ".jpg"))
    ax1.set_title('Style image 1', y=-0.75)
    ax2.imshow(img.imread("images/" + style_image_name21 + ".jpg"))
    ax2.set_title('Style image 2', y=-0.75)
    ax3.imshow(img.imread("images/" + content_image_name1 + ".jpg"))
    ax3.set_title('Content image', y=-0.4)
    ax4.imshow(image)
    ax4.set_title('Output')
    plt.show()
    return image


# Функция для отображения изображения
def show_image(data, title=None):
    image = data.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


# Функция, выполняющая трансформацию стиля
def do_transfer2(content_image_name1, style_image_name11, style_weight1, content_weight1):
    content_image = load_image("images/" + content_image_name1, type1='content')
    style_image1 = load_image("images/" + style_image_name11)
    # style_image2 = load_image("images/" + style_image_name21)
    input_img = content_image.clone()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_image, style_image1, input_img,
                                num_steps=n_iter, style_weight1=style_weight1, content_weight1=content_weight1)
    return output
    # return make_global_image(output, content_image_name1, style_image_name11, style_image_name21)


def make_style_transfer():
    result_image = do_transfer2(content_image_name, style_image_name,
                                style_weight, content_weight)
    # show_image(result_image)
    unloader(result_image.cpu().clone().squeeze(0)).save(fp=f'images/result.jpg')
