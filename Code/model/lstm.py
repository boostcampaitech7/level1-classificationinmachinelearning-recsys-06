import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Model, self).__init__()
        # input_size: features 컬럼 수만큼 입력
        # output_size : 결과값 Colum 수
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.tensor:
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력을 사용
        return out

    def predict(self, x) -> torch.tensor:
        return self.forward(x)

    def fit(self, x_train: torch.tensor, y_train: torch.tensor):
        from torch.utils.data import DataLoader, TensorDataset

        # DataLoader 준비
        train_dataset = TensorDataset(x_train.unsqueeze(1), y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

        # 모델 초기화

        model = Model(
            self.input_size, self.hidden_size, self.output_size, self.num_layers
        )
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 학습
        num_epochs = 100
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
