using TensorFlow, Distributions, MLDataUtils, MLDatasets
using DataFrames, CSV
using Plots
pyplot()

const display_step = 10
const batch_size = 50

const n_input = 1
const n_steps = 240
const n_hidden = 128
const n_classes = 1

nikkeis_raw = CSV.read("./n225.csv")
# nikkeis_raw = [fill(float(get(r)), 1, 1) for r in nikkeis_raw[:close]]
nikkeis_raw = convert(Array{Float64, 1}, nikkeis_raw[:close])
const train_data = BatchView(nikkeis_raw[1:10000], batch_size)
const train_labels = [fill(r[end], 1, 1) for r in train_data]

const test_data = BatchView(nikkeis_raw[10001:12000], batch_size)
const test_labels = [fill(r[end], 1, 1) for r in test_data]

function training()
  kk = 0
  preds = []

  for _ in 1:1
    for (xs_a, ys_a) in zip(train_data, train_labels)
      # xs = cat(3, xs_a)
      xs = zeros(50, 1)
      for i in 1:50
        xs[i] = xs_a[i]
      end
      ys = collect(ys_a)
      _, pred = run(sess, [optimizer, Y_pred], Dict(X=>xs, Y_obs=>ys))
      info("pred: $(pred), obs: $(ys)")
      push!(preds, pred)

      kk += 1
      if kk % display_step == 1
        train_accuracy, train_cost, y, h, s, w = run(sess, [accuracy, cost, Y_pred, Hs, status, W], Dict(X=>xs, Y_obs=>ys))
        info("step $(kk), loss = $(train_cost), accuracy $(train_accuracy)")
        # info("Y_pred $(y),\nHs[end] $(h[end]),\nstatus $(s),\nW $(w)")
      end

      summaries = run(sess, merged_summary_op, Dict(X=>xs, Y_obs=>ys))
      write(summary_writer, summaries, kk)
    end
  end

  train_labels_ = [r[1] for r in train_labels]
  preds_ = [r[1] for r in preds]
  plot(1:length(preds), [preds_, train_labels_], label = ["prediction" "label"])
  savefig("traininghidden1283times.png")
end

function testing()
  test_cost_summary = TensorFlow.summary.histogram("TestCost", cost)
  test_accuracy_summary = TensorFlow.summary.histogram("TestAccuracy", accuracy)
  test_merged_summary_op = TensorFlow.summary.merge_all()

  t_preds = []

  for _ in 1:1
    for (xs_a, ys_a) in zip(test_data, test_labels)
      # xs = cat(3, xs_a)
      xs = zeros(50, 1)
      for i in 1:50
        xs[i] = xs_a[i]
      end
      ys = collect(ys_a)
      _, test_accuracy, t_pred = run(sess, [optimizer, accuracy, Y_pred], Dict(X=>xs, Y_obs=>ys))
      info("pred: $(t_pred), obs: $(ys), accuracy: $(test_accuracy)")
      push!(t_preds, t_pred)

      summaries = run(sess, test_merged_summary_op, Dict(X=>xs, Y_obs=>ys))
      write(summary_writer, summaries)
    end
  end

  test_labels_ = [r[1] for r in test_labels]
  # t_preds_ = collect(Float64, t_preds)
  t_preds_ = [r[1] for r in t_preds]
  plot(1:length(t_preds), [t_preds_, test_labels_], label = ["prediction" "label"])
  savefig("testhidden1283times.png")
end

sess = Session(Graph())

X = placeholder(Float32, shape = [50, 1], name="X")
# x = transpose(X, Int32.([1, 3, 2].-1)) # Permuting batch_size and n_steps. (the -1 is to use 0 based indexing)
# x = reshape(X, [50, 1]) # Reshaping to (n_steps*batch_size, n_input)
x = split(1, 50, X) # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
Y_obs = placeholder(Float32, shape = [1, 1], name="Y_obs")

variable_scope("model", initializer=Normal(0, 0.5)) do
  global W = get_variable("weights", [n_hidden, 1], Float32)
  global b = get_variable("bias", [1], Float32)
end

Hs, status = nn.rnn(nn.rnn_cell.LSTMCell(n_hidden), x, dtype=Float32)

# HsかWがおかしい → softmax使ってたのがいけなかった(softmaxの定義からY_predが1になるのは自明)
Y_pred = Hs[end] * W + b # Y_pred: shape=(1, 1) Y_predはFloat32(日経平均)

# Y_obsがおかしい
# cost = reduce_mean(-reduce_sum(Y_obs.*log(Y_pred), axis=[2])) # 30行目のコメントより、Y_obsもFloat32(正しい(というか予測させたい)日経平均)
cost = reduce_sum((Y_obs .- Y_pred)^2) + nn.l2_loss(W)

optimizer = train.minimize(train.GradientDescentOptimizer(), cost)

# 精度の評価がおかしい
correct_prediction = Y_pred / Y_obs

accuracy = reduce_mean(cast(correct_prediction, Float32))

cost_summary = TensorFlow.summary.histogram("Cost", cost)
accuracy_summary = TensorFlow.summary.histogram("Accuracy", accuracy)
merged_summary_op = TensorFlow.summary.merge_all()
summary_writer = TensorFlow.summary.FileWriter("./log")

run(sess, global_variables_initializer())

training()
testing()
