# このコードはJulia 1.93で動作します。
# https://leadinge.co.jp/rd/2021/05/31/818/
# Juliaで深層学習 Vol.2
# 2023/9/24 河野保　一部修正

using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux:throttle
using Flux.Losses: logitcrossentropy
using MLDatasets
using ImageShow

#学習データ 注）変更あり
train_x,train_y=MLDatasets.MNIST(:train)[:]
#テストデータ　注）変更あり
test_x,test_y=MLDatasets.MNIST(:test)[:]

#　追加要
train_x = Flux.flatten(train_x) # 784×60000
test_x = Flux.flatten(test_x) # 784×10000
#下記でもOK
#train_x = Flux.unsqueeze(train_x, 3)
#test_x = Flux.unsqueeze(test_x, 3)

# One-hot-encode the labels
train_y = onehotbatch(train_y, 0:9) # 10×60000
test_y = onehotbatch(test_y, 0:9) # 10×10000

batch_size = 128
#shuffle=true

# 変更あり　DataLoader(...,batchsize=128,shuffle =true)
# → DataLoader(...,batchsize=128)

train_data = DataLoader((train_x, train_y), batchsize=128)
test_data = DataLoader((test_x, test_y), batchsize=128)

#手書き数字は28X28サイズの画像データとなっている
MLDatasets.convert2image(MNIST,MLDatasets.MNIST(split=:train).features[:,:,1])
#下記コードは deprecated error
#MLDatasets.MNIST.convert2image(MLDatasets.MNIST.traintensor(1))

#　モデル定義
model = Chain(
#flatten,
Dense(28^2, 32, relu),
Dropout(0.2),
Dense(32, 10))

#バラメータ
params = Flux.params(model)

#最適化関数
optimiser=ADAM()

#損失関数
loss(x,y)=logitcrossentropy(model(x),y)

#コールバック関数
evalcb = () -> @show(loss(train_x,train_y))

#学習
for i in 1:10
    Flux.train!(loss, params, train_data, optimiser, cb = throttle(evalcb, 5))
end

#正解率の算出

function accuracy(data_loader, model)
    acc_correct = 0
    for (x, y) in data_loader
        batch_size = size(x)[end]
        acc_correct += sum(onecold(model(x)) .== onecold(y)) / batch_size
    end
    return acc_correct / length(data_loader)
end

#モデルをテストモードの変更

testmode!(model)
@show accuracy(test_data,model)

