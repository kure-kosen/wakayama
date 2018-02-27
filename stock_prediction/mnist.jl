using TensorFlow
using MNIST
using MLDataUtils
using MLLabelUtils

const batch_size = 100

image_train, label_train = traindata()
trainlabels = BatchView(convertlabel(LabelEnc.OneOfK{Float32}, label_train), batch_size)
traindatum = BatchView(image_train, batch_size)

image_test, label_test = testdata()
testlabels = BatchView(convertlabel(LabelEnc.OneOfK{Float32}, label_test), batch_size)
testdatum = BatchView(image_test, batch_size)

x = placeholder(Float32, shape=[784, batch_size], name="x") # MNISTの画像が入るプレースホルダー
W = Variable(zeros(784, 10), name="W") # 重み
b = Variable(zeros(10), name="b") # バイアス
y = nn.softmax(x * W + b) # モデル

y_ = placeholder(Float32, shape=[10, batch_size], name="y_")
cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))

# train_step = train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
optimizer = train.minimize(train.AdamOptimizer(0.5), cross_entropy)

sess = Session(Graph())
run(sess, global_variables_initializer())

for (xs_a, ys_a) in zip(traindatum, trainlabels)
  xs = collect(Float32, xs_a)
  ys = collect(Float32, ys_a)
  run(sess, optimizer, Dict(x=>xs, y_=>ys))
end

correct_prediction = equal(argmax(y, 1), argmax(y_ 1))
accuracy = reduce_mean(cast(correct_prediction, Float32))
# run(sess, accuracy, Dict(x=>image_test, y_=>label_test))

batch_accuracies = []
for (ii, (xs_a, ys_a)) in enumerate(zip(testdatum, testlabels))
    xs = collect(Float32, xs_a)
    ys = collect(Float32, ys_a)

    batch_accuracy = run(sess, accuracy, Dict(x=>xs, y_=>ys))
    push!(batch_accuracies, batch_accuracy)
end
@show mean(batch_accuracies) #Mean of means of consistantly sized batchs is the overall mean
