import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
#class CNN(nn.Module):#
#	def __init__(self, arch, channels, kernels, out, input_shape, # cnn parameters
#				 hidden_dims=[], activation='relu', p_drop=0, batchnorm='False' # mlp parameters#
#				 ):
#		super(CNN, self).__init__()#
#		self.features = nn.Sequential()

#		i, j , l = 0, 0, 0
#		for layer in arch:
#			if layer == 'conv':
#				self.features.add_module(
#				name=str(l),
#				module=nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[j], stride=1, padding=1))
#				i += 1
#				j += 1
#			elif layer == 'maxpool':
#				self.features.add_module(
#				name=str(l),
#				module=nn.MaxPool2d(kernel_size=kernels[j], stride=2, padding=0))
#				j +=1
#			else:
#				self.features.add_module(
#				name=str(l),
#				module=nn.ReLU(inplace=True) if layer=='relu' else 
#					  (nn.Tanh() if layer=='tanh' else nn.Sigmoid()))    
#			l += 1
#
#		flt_dim = self.flattendim(input_shape)
#		self.linear = MLP(flt_dim, out, hidden_dims, activation, p_drop, batchnorm) # number 128 needs to be found
 #   
#	def forward(self, x):
#		x = self.features(x)
#		#         return x
#		x = x.view((len(x), -1))
#		return self.linear(x)
#
#	def flattendim(self, input_shape):
#		for module in self.features:
#			name = module.__class__.__name__
#			if name == 'Conv2d':
#				(cin, hin, win) = input_shape
#				cout = module.out_channels
#				hout = int(np.floor((hin + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1))
#				wout = int(np.floor((win + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) / module.stride[1] + 1))
#				input_shape = (cout, hout, wout)
#			elif name == 'MaxPool2d':
#				(cin, hin, win) = input_shape
#				cout = cin
#				hout = int(np.floor((hin + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) / module.stride + 1))
#				wout = int(np.floor((win + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) / module.stride + 1))
#				input_shape = (cout, hout, wout)
#
#		return int(np.prod(np.array(input_shape)))

