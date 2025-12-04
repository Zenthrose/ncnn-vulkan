#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QListWidget>
#include <QProgressBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QUrl>
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include "llm_engine.h"
#include "../examples/tool.h"

class ChatWindow : public QMainWindow {
    Q_OBJECT

public:
    ChatWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("NCNN LLM Chat");
        setGeometry(100, 100, 800, 600);

        QWidget *central = new QWidget;
        setCentralWidget(central);

        QVBoxLayout *layout = new QVBoxLayout(central);

        // Model list
        modelList = new QListWidget;
        layout->addWidget(modelList);

        QHBoxLayout *modelButtons = new QHBoxLayout;
        loadModelButton = new QPushButton("Load Model");
        downloadModelButton = new QPushButton("Download Model");
        deleteModelButton = new QPushButton("Delete Model");
        modelButtons->addWidget(loadModelButton);
        modelButtons->addWidget(downloadModelButton);
        modelButtons->addWidget(deleteModelButton);
        layout->addLayout(modelButtons);

        // Chat area
        chatArea = new QTextEdit;
        chatArea->setReadOnly(true);
        layout->addWidget(chatArea);

        // Input
        QHBoxLayout *inputLayout = new QHBoxLayout;
        inputLine = new QLineEdit;
        sendButton = new QPushButton("Send");
        inputLayout->addWidget(inputLine);
        inputLayout->addWidget(sendButton);
        layout->addLayout(inputLayout);

        // Progress
        progressBar = new QProgressBar;
        progressBar->setVisible(false);
        layout->addWidget(progressBar);

        // Settings button
        settingsButton = new QPushButton("Settings");
        layout->addWidget(settingsButton);

        // Status
        statusBar = new QStatusBar;
        setStatusBar(statusBar);

        // Network
        network = new QNetworkAccessManager(this);

        // LLM Engine
        engine = new ncnn::LLMEngine;

        // Connect signals
        connect(loadModelButton, &QPushButton::clicked, this, &ChatWindow::loadModel);
        connect(downloadModelButton, &QPushButton::clicked, this, &ChatWindow::downloadModel);
        connect(deleteModelButton, &QPushButton::clicked, this, &ChatWindow::deleteModel);
        connect(sendButton, &QPushButton::clicked, this, &ChatWindow::sendMessage);
        connect(settingsButton, &QPushButton::clicked, this, &ChatWindow::showSettings);

        // Load local models
        loadLocalModels();
    }

private slots:
    void loadModel() {
        QString fileName = QFileDialog::getOpenFileName(this, "Load Model", "", "GGUF Files (*.gguf)");
        if (!fileName.isEmpty()) {
            progressBar->setVisible(true);
            progressBar->setRange(0, 0); // indeterminate
            statusBar->showMessage("Loading model...");
            // Load synchronously for simplicity
            if (engine->load_model(fileName.toStdString())) {
                modelList->addItem(fileName);
                statusBar->showMessage("Model loaded");
            } else {
                QMessageBox::warning(this, "Error", "Failed to load model");
            }
            progressBar->setVisible(false);
        }
    }

    void downloadModel() {
        QString url = QInputDialog::getText(this, "Download Model", "Enter URL:");
        if (!url.isEmpty()) {
            QUrl qurl(url);
            QNetworkRequest request(qurl);
            QNetworkReply *reply = network->get(request);
            connect(reply, &QNetworkReply::downloadProgress, this, &ChatWindow::downloadProgress);
            connect(reply, &QNetworkReply::finished, this, [this, reply]() {
                if (reply->error() == QNetworkReply::NoError) {
                    QString fileName = QFileDialog::getSaveFileName(this, "Save Model", "", "GGUF Files (*.gguf)");
                    if (!fileName.isEmpty()) {
                        QFile file(fileName);
                        if (file.open(QIODevice::WriteOnly)) {
                            file.write(reply->readAll());
                            file.close();
                            modelList->addItem(fileName);
                            statusBar->showMessage("Model downloaded");
                        }
                    }
                } else {
                    QMessageBox::warning(this, "Error", "Download failed: " + reply->errorString());
                }
                progressBar->setVisible(false);
                reply->deleteLater();
            });
            progressBar->setVisible(true);
            progressBar->setRange(0, 100);
        }
    }

    void deleteModel() {
        QListWidgetItem *item = modelList->currentItem();
        if (item) {
            QString fileName = item->text();
            if (QMessageBox::question(this, "Delete Model", "Delete " + fileName + "?") == QMessageBox::Yes) {
                QFile::remove(fileName);
                delete item;
                statusBar->showMessage("Model deleted");
            }
        }
    }

    void sendMessage() {
        QString text = inputLine->text();
        if (text.isEmpty()) return;
        chatArea->append("You: " + text);
        inputLine->clear();
        // Generate response
        try {
            std::string response = engine->generate_text(text.toStdString(), config);
            chatArea->append("Bot: " + QString::fromStdString(response));
        } catch (const std::exception& e) {
            QMessageBox::warning(this, "Error", "Generation failed: " + QString(e.what()));
        }
    }

    void showSettings() {
        // Simple settings
        bool ok;
        double temp = QInputDialog::getDouble(this, "Settings", "Temperature:", config.temperature, 0.0, 2.0, 2, &ok);
        if (ok) config.temperature = temp;
        int max_tokens = QInputDialog::getInt(this, "Settings", "Max Tokens:", config.max_tokens, 1, 10000, 1, &ok);
        if (ok) config.max_tokens = max_tokens;
    }

    void downloadProgress(qint64 bytesReceived, qint64 bytesTotal) {
        if (bytesTotal > 0) {
            progressBar->setValue((bytesReceived * 100) / bytesTotal);
        }
    }

private:
    void loadLocalModels() {
        QStringList dirs = QStandardPaths::standardLocations(QStandardPaths::HomeLocation);
        if (!dirs.isEmpty()) {
            QDir dir(dirs.first());
            QStringList filters;
            filters << "*.gguf";
            QStringList files = dir.entryList(filters, QDir::Files);
            for (const QString& file : files) {
                modelList->addItem(dir.absoluteFilePath(file));
            }
        }
    }

    QListWidget *modelList;
    QPushButton *loadModelButton, *downloadModelButton, *deleteModelButton;
    QTextEdit *chatArea;
    QLineEdit *inputLine;
    QPushButton *sendButton;
    QProgressBar *progressBar;
    QStatusBar *statusBar;
    QPushButton *settingsButton;
    QNetworkAccessManager *network;
    ncnn::LLMEngine *engine;
    ncnn::GenerationConfig config;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    ChatWindow window;
    window.show();
    return app.exec();
}