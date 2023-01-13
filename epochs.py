import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

loss_colors = ["#88e99a", "#02531d"]

def q(field):

    ret = field.split(":")[1]

    return float(ret)
    
def parse(_file):

    reader = open(_file, "r")
    lines = reader.readlines()
    reader.close()

    train_losses = []
    valid_losses = []
    train_acc1 = []
    valid_acc1 = []
    train_acc3 = []
    valid_acc3 = []
    train_acc5 = []
    valid_acc5 = []
    
    
    for line in lines:
        if(line.find("Epoch") != -1):
            fields = line.split("|")
            train_losses.append(q(fields[1]))
            valid_losses.append(q(fields[2]))
            train_acc1.append(q(fields[3]))
            valid_acc1.append(q(fields[4]))
            train_acc3.append(q(fields[5]))
            valid_acc3.append(q(fields[6]))
            train_acc5.append(q(fields[7]))
            valid_acc5.append(q(fields[8]))
            
    return train_losses, valid_losses, train_acc1, valid_acc1, train_acc3, valid_acc3, train_acc5, valid_acc5



if __name__ == "__main__":

    #_file = "from_scratch_fam_sgd.txt"
    _file = sys.argv[1]
    _title = sys.argv[2]
    
    train_losses, valid_losses, train_acc1, valid_acc1, train_acc3, valid_acc3, train_acc5, valid_acc5 = parse(_file)

    print("Train Losses:", train_losses)
    print("Validation Losses:", valid_losses)

    
    xx = []
    for i in range(1,101):
        xx.append(i)
        
    #fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x = xx,
                             y = train_acc1,
                             line = dict(color='darkblue', width=4),
                             name = "Train Accuracy Top 1"),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x = xx,
                             y = train_acc3,
                             line = dict(color='darkblue', width=4, dash='dash'),
                             name = "Train Accuracy Top 3"),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x = xx,
                             y = train_acc5,
                             line = dict(color='darkblue', width=4, dash='dashdot'),
                             name = "Train Accuracy Top 5"),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x = xx,
                             y = valid_acc1,
                             line = dict(color='firebrick', width=4),
                             name = "Validation Accuracy Top 1"),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x = xx,
                             y = valid_acc3,
                             line = dict(color='firebrick', width=4, dash='dash'),
                             name = "Validation Accuracy Top 3"),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x = xx,
                             y = valid_acc5,
                             line = dict(color='firebrick', width=4, dash='dashdot'),
                             name = "Validation Accuracy Top 5"),
                  secondary_y=False)
    
    
    fig.add_trace(go.Scatter(x = xx,
                             y = train_losses,
                             line = dict(color=loss_colors[0], width=4),
                             name = "Train Loss"),
                  secondary_y = True)

    fig.add_trace(go.Scatter(x = xx,
                             y = valid_losses,
                             line = dict(color=loss_colors[1], width=4),
                             name = "Validation Loss"),
                  secondary_y = True)


    fig.update_layout(template="simple_white")
    fig.update_layout(
        yaxis2=dict(
        title="Loss",
        titlefont=dict(
            color="#42952e"
        ),
        tickfont=dict(
            color="#42952e"
        ),
        anchor="x",
        overlaying="y",
        side="right"
    )
    )

    fig.update_layout(
        title=_title,
        font=dict(
            family="Courier New, monospace",
            size=22,
            color="black"
        )
    )
    
    fig.update_layout(legend=dict(yanchor="top",y=1.12,xanchor="left",x=0.01,orientation="h"),
                      margin=dict(l=20, r=5, t=170, b=10))
    fig.layout['yaxis'].update(title="Accuracy [%]")
    fig.layout['xaxis'].update(title="Epochs")
    
    fig.show()
