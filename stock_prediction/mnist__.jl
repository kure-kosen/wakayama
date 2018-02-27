using TensorFlow
using Distributions
using ProgressMeter
using MLLabelUtils
using MLDataUtils
using MLDatasets
using Base.Test

#Training Hyper Parameter
const learning_rate = 0.001
const training_iters = 2 #Just two, becuase I don't have anything to stop overfitting and I don't got all day
const batch_size = 256
const display_step = 100 #How often to display the 

# Network Parameters
const n_input = 28 # MNIST data input (img shape: 28*28)
const n_steps = 28 # timesteps
const n_hidden = 128 # hidden layer num of features
const n_classes = 10; # MNIST total classes (0-9 digits)

encode(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
const traindata_raw,trainlabels_raw = MNIST.traindata();
const trainlabels = BatchView(encode(trainlabels_raw), batch_size)
const traindata = BatchView(traindata_raw, batch_size);

sess = Session(Graph())
X = placeholder(Float32, shape=[n_steps, n_input, batch_size])
Y_obs = placeholder(Float32, shape=[n_classes, batch_size])

variable_scope("model", initializer=Normal(0, 0.5)) do
  global W = get_variable("weights", [n_hidden, n_classes], Float32)
  global B = get_variable("bias", [n_classes], Float32)
end

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (n_steps, n_input, batch_size) from the way we declared X (and the way the data actually comes)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

x = transpose(X, Int32.([1, 3, 2].-1)) # Permuting batch_size and n_steps. (the -1 is to use 0 based indexing)
x = reshape(x, [n_steps*batch_size, n_input]) # Reshaping to (n_steps*batch_size, n_input)
x = split(1, n_steps, x) # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)

Hs, states = nn.rnn(nn.rnn_cell.LSTMCell(n_hidden), x; dtype=Float32);
Y_pred = nn.softmax(Hs[end]*W + B)

cost = reduce_mean(-reduce_sum(Y_obs.*log(Y_pred), axis=[1])) #cross entropy

optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)

correct_prediction = indmax(Y_obs, 2) .== indmax(Y_pred, 2)
accuracy = reduce_mean(cast(correct_prediction, Float32))

run(sess, global_variables_initializer())

kk=0
for jj in 1:training_iters
  for (xs_a, ys_a) in zip(traindata, trainlabels)
    @show xs_a
    @show collect(xs_a)
    xs = collect(xs_a)
    ys = collect(ys_a)
    run(sess, optimizer,  Dict(X=>xs, Y_obs=>ys))
    kk+=1
    if kk % display_step == 1
      train_accuracy, train_cost = run(sess, [accuracy, cost], Dict(X=>xs, Y_obs=>ys))
      info("step $(kk*batch_size), loss = $(train_cost),  accuracy $(train_accuracy)")
    end
  end
end

testdata_raw, testabels_raw = MNIST.testdata()
testlabels = BatchView(encode(testabels_raw), batch_size)
testdata = BatchView(testdata_raw, batch_size);


batch_accuracies = []
for (ii, (xs_a, ys_a)) in enumerate(zip(testdata, testlabels))
    xs = collect(xs_a)
    ys = collect(ys_a)'

    batch_accuracy = run(sess, accuracy, Dict(X=>xs, Y_obs=>ys))
    #info("step $(ii),   accuracy $(batch_accuracy )")
    push!(batch_accuracies, batch_accuracy)
end
@show mean(batch_accuracies) #Mean of means of consistantly sized batchs is the overall mean
