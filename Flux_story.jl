# このコードはJulia 1.93で動作します。
# 元コード　https://towardsdatascience.com/deep-learning-with-julia-flux-jl-story-7544c99728ca
# 河野保　一部修正
# 2023/9/24 河野保　確認

using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy,throttle
# using Flux: @epochs // depricated
using Statistics
using MLDatasets
# Load the data

x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_valid, y_valid = MLDatasets.MNIST(split=:test)[:]
# Add the channel layer
x_train = Flux.unsqueeze(x_train, 3)
x_valid = Flux.unsqueeze(x_valid, 3)
# Encode labels
y_train = onehotbatch(y_train, 0:9)
y_valid = onehotbatch(y_valid, 0:9)
# Create the full dataset
#河野注　DataLoader第一引数はtapleか一つのテンソール
#train_data = DataLoader(x_train, y_train, batchsize=128)
train_data = DataLoader((x_train, y_train), batchsize=128)

model = Chain(
    # 28x28 => 14x14
    Conv((5, 5), 1=>8, pad=2, stride=2, relu),
    # 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=1, stride=2, relu),
    # 7x7 => 4x4
    Conv((3, 3), 16=>32, pad=1, stride=2, relu),
    # 4x4 => 2x2
    Conv((3, 3), 32=>32, pad=1, stride=2, relu),
    
    # Average pooling on each width x height feature map
    GlobalMeanPool(),
    Flux.flatten,
    
    Dense(32, 10),
    softmax)


    # Getting predictions
ŷ = model(x_train)
# Decoding predictions
ŷ = onecold(ŷ)
println("Prediction of first image: $(ŷ[1])")

accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))
loss(x, y) = Flux.crossentropy(model(x), y)
# learning rate
lr = 0.1
opt = Descent(lr)
ps = Flux.params(model)

#コールバック関数
evalcb = () -> @show(loss(x_train, y_train))

number_epochs = 10
#@epochs number_epochs Flux.train!(loss, ps, train_data, opt)
for epoch in 1:10
  Flux.train!(loss, ps,train_data, opt,cb = throttle(evalcb, 5)) 
end


@show accuracy(model(x_train), y_train)

@show accuracy(model(x_valid), y_valid)
