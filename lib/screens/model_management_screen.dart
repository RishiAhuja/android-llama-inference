import 'package:flutter/material.dart';
import '../services/model_manager.dart';

class ModelManagementScreen extends StatefulWidget {
  const ModelManagementScreen({super.key});

  @override
  State<ModelManagementScreen> createState() => _ModelManagementScreenState();
}

class _ModelManagementScreenState extends State<ModelManagementScreen> {
  final ModelManager _modelManager = ModelManager();
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _checkModelStatus();
  }

  Future<void> _checkModelStatus() async {
    setState(() => _isLoading = true);
    await _modelManager.checkModelStatus();
    setState(() => _isLoading = false);
  }

  Future<void> _downloadModel() async {
    await _modelManager.downloadModel(
      onProgress: (progress) {
        setState(() {});
      },
      onError: (error) {
        setState(() {});
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $error')),
        );
      },
    );
    setState(() {});
  }

  Future<void> _deleteModel() async {
    await _modelManager.deleteModel();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Management'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Gemma 3B Model',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 8),
                    Text('Size: ~800MB'),
                    const SizedBox(height: 8),
                    _buildStatusWidget(),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            _buildActionButtons(),
            const Spacer(),
            if (_modelManager.status == ModelStatus.downloaded)
              ElevatedButton(
                onPressed: () => Navigator.pushReplacementNamed(context, '/chat'),
                child: const Text('Start Chatting'),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusWidget() {
    if (_isLoading) {
      return const Row(
        children: [
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(width: 8),
          Text('Checking status...'),
        ],
      );
    }

    switch (_modelManager.status) {
      case ModelStatus.notDownloaded:
        return const Row(
          children: [
            Icon(Icons.cloud_download, color: Colors.grey),
            SizedBox(width: 8),
            Text('Not downloaded'),
          ],
        );
      case ModelStatus.downloading:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.downloading, color: Colors.blue),
                SizedBox(width: 8),
                Text('Downloading...'),
              ],
            ),
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _modelManager.downloadProgress),
            const SizedBox(height: 4),
            Text('${(_modelManager.downloadProgress * 100).toInt()}%'),
          ],
        );
      case ModelStatus.downloaded:
        return const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green),
            SizedBox(width: 8),
            Text('Downloaded and ready'),
          ],
        );
      case ModelStatus.error:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.error, color: Colors.red),
                SizedBox(width: 8),
                Text('Error'),
              ],
            ),
            if (_modelManager.errorMessage.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  _modelManager.errorMessage,
                  style: const TextStyle(color: Colors.red, fontSize: 12),
                ),
              ),
          ],
        );
    }
  }

  Widget _buildActionButtons() {
    switch (_modelManager.status) {
      case ModelStatus.notDownloaded:
      case ModelStatus.error:
        return ElevatedButton.icon(
          onPressed: _downloadModel,
          icon: const Icon(Icons.download),
          label: const Text('Download Model'),
        );
      case ModelStatus.downloading:
        return ElevatedButton.icon(
          onPressed: null,
          icon: const SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          label: const Text('Downloading...'),
        );
      case ModelStatus.downloaded:
        return Column(
          children: [
            ElevatedButton.icon(
              onPressed: _downloadModel,
              icon: const Icon(Icons.refresh),
              label: const Text('Re-download'),
            ),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: _deleteModel,
              icon: const Icon(Icons.delete),
              label: const Text('Delete Model'),
            ),
          ],
        );
    }
  }
}
