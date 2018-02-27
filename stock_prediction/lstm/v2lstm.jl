using TensorFlow, Distributions
using DataFrames, CSV
using Plots;pyplot()

const path = joinpath("./", ARGS[3])
const length_of_sequence = 50 # 食わせたいデータの数
const num_of_hidden_nodes = parse(Int64, ARGS[3])
const num_of_inputs = 1 # インプットの種類(今回だと終値だけなので1)
const num_of_outputs = 1 # アウトプットの種類(今回は出力される終値なので1)
const display_step = 100

function batch_data(raw_data::Array{Float64, 1})
  const last_index = length(raw_data) - length_of_sequence

  data, labels = [], []

  for i in 1:last_index
    push!(data, raw_data[i:(length_of_sequence + (i - 1))])
    push!(labels, raw_data[length_of_sequence + i])
  end

  data = [cat(2, r) for r in data]
  labels = [cat(2, r) for r in labels]

  return data, labels
end

function get_data(file_name, column_name, ratio=0.8)
  raw_data = CSV.read(file_name)[Symbol(column_name)]
  raw_data = [get(r) for r in collect(raw_data)]

  divide_index = Int(round(length(raw_data) * ratio))
  train_data, train_labels = batch_data(raw_data[1:divide_index])
  test_data, test_labels = batch_data(raw_data[divide_index:end])

  return train_data, train_labels, test_data, test_labels # train_data: 50 x 1 x n の配列, train_labels: 1 x n の配列, test_data, test_labels: train_data, trainlabelsに同じ
end

function training(datum, labels)
  kk = 0
  preds = []
  losses = []

  for (data, label) in zip(datum, labels)
    _, pred, loss = run(sess, [train_op, Y_pred, loss_op], Dict(X=>data, Y_obs=>label))
    push!(preds, pred)
    push!(losses, loss)

    kk += 1
    if kk % display_step == 1
      train_loss, train_pred = run(sess, [loss_op, Y_pred], Dict(X=>data, Y_obs=>label) )
      info("n: $(kk)\nloss: $(train_loss)\nprediction: $(train_pred)\n")
    end

    summaries = run(sess, summary_op, Dict(X=>data, Y_obs=>label))
    write(summary_writer, summaries, kk)
  end

  train_labels_ = [r[1] for r in labels]
  preds_ = [r[1] for r in preds]
  plot(1:length(preds), [preds_, train_labels_], label = ["prediction" "label"])
  savefig(joinpath(path, "train_pred_label.png"))

  losses_ = [r[1] for r in losses]
  plot(1:length(losses), [losses], label = ["loss"])
  savefig(joinpath(path, "train_loss.png"))

  plot(1:length(preds), [preds_, train_labels_, losses_], label = ["prediction" "label" "losses"])
  savefig(joinpath(path, "train.png"))
end

function testing(datum, labels)
  kk = 0
  preds = []
  losses = []

  for (data, label) in zip(datum, labels)
    kk += 1
    if kk % display_step == 1
      test_loss, test_pred = run(sess, [loss_op, Y_pred], Dict(X=>data, Y_obs=>label) )
      info("n: $(kk)\nloss: $(test_loss)\nprediction: $(test_pred)\n")
    end

    _, summaries, pred, loss = run(sess, [train_op, summary_op, Y_pred, loss_op], Dict(X=>data, Y_obs=>label))
    push!(preds, pred)
    push!(losses, loss)

    write(summary_writer, summaries, kk)
  end

  test_labels_ = [r[1] for r in labels]
  # preds_ = collect(Float64, preds)
  preds_ = [r[1] for r in preds]
  plot(1:length(preds), [preds_, test_labels_], label = ["prediction" "label"])
  savefig(joinpath(path, "test_pred_label.png"))

  losses_ = [r[1] for r in losses]
  plot(1:length(losses), [losses], label = ["loss"])
  savefig(joinpath(path, "test_loss.png"))

  plot(1:length(preds), [preds_, test_labels_, losses_], label = ["prediction" "label" "losses"])
  savefig(joinpath(path, "test.png"))
end

sess = Session(Graph())

# データ整形
# 
# テンソル定義(shape=[]のテンソルをデータの個数もつ配列)
#   LSTMに入れるべきテンソルのランクがわからない ← 一回の入力×食わせたいデータの個数(隠れ層の個数? 隠れ層は各セルに存在する?)
X = placeholder(shape=[length_of_sequence, num_of_inputs], Float64, name="X")
x = split(1, length_of_sequence, X)

cell = nn.rnn_cell.LSTMCell(num_of_hidden_nodes)
output, state = nn.rnn(cell, x, dtype=Float32) # dtype: 出力の型

# 最初から何かしらの値を入れておくと収束が早いかも?
variable_scope("model", initializer=Normal(0, 0.5)) do
  global W = get_variable("weights", [num_of_hidden_nodes, num_of_inputs], Float64)
  global b = get_variable("bias", [num_of_outputs], Float64)
end

Y_pred = output[end] * W + b
Y_obs = placeholder(shape=[num_of_inputs, num_of_outputs], Float64, name="Y_obs")
# Y_obs = placeholder(shape=[], Float64, name="Y_obs")

W_hist = TensorFlow.summary.histogram("W_hist", W)
b_hist = TensorFlow.summary.histogram("b_hist", b)
Y_pred_hist = TensorFlow.summary.histogram("Y_pred", Y_pred)

loss_op = reduce_mean((Y_pred - Y_obs)^2)
loss_op_hist = TensorFlow.summary.scalar("loss", loss_op)

train_op = train.minimize(train.GradientDescentOptimizer(), loss_op)

summary_op = TensorFlow.summary.merge_all()
summary_writer = TensorFlow.summary.FileWriter(path)

run(sess, global_variables_initializer())

train_data, train_labels, test_data, test_labels = get_data(ARGS[1], ARGS[2])
training(train_data, train_labels)
testing(test_data, test_labels)
