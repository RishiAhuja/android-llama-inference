import 'package:flutter/material.dart';
import '../models/chat_message.dart';
import '../services/llama_service.dart';
import '../services/model_manager.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final LlamaService _llamaService = LlamaService();
  final ModelManager _modelManager = ModelManager();
  final TextEditingController _promptController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];

  bool _isLoading = true;
  bool _isGenerating = false;

  @override
  void initState() {
    super.initState();
    _initializeChat();
  }

  Future<void> _initializeChat() async {
    setState(() {
      _isLoading = true;
      _messages.add(ChatMessage(text: "Initializing...", isUser: false));
    });

    await _modelManager.checkModelStatus();

    if (_modelManager.status != ModelStatus.downloaded) {
      setState(() {
        _messages.last = ChatMessage(
          text:
              "Model not found. Please go to Model Management to download it.",
          isUser: false,
        );
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _messages.last = ChatMessage(text: "Loading model...", isUser: false);
    });

    final success =
        await _llamaService.loadModel(_modelManager.modelFile!.path);

    setState(() {
      if (success) {
        _messages.last = ChatMessage(
          text: "Model loaded! Ask me anything.",
          isUser: false,
        );
      } else {
        _messages.last = ChatMessage(
          text: "Failed to load model. Please try restarting the app.",
          isUser: false,
        );
      }
      _isLoading = false;
    });
  }

  Future<void> _sendMessage() async {
    if (_promptController.text.trim().isEmpty ||
        _isGenerating ||
        !_llamaService.isInitialized) {
      return;
    }

    final prompt = _promptController.text.trim();
    _promptController.clear();

    setState(() {
      _messages.add(ChatMessage(text: prompt, isUser: true));
      _messages.add(ChatMessage(text: "Thinking...", isUser: false));
      _isGenerating = true;
    });

    _scrollToBottom();

    final stopwatch = Stopwatch()..start();

    try {
      final response = await _llamaService.generateResponse(prompt);
      stopwatch.stop();
      
      setState(() {
        // Replace the "Thinking..." message with the actual response including timing
        _messages.last = ChatMessage(
          text: response, 
          isUser: false,
          responseTime: stopwatch.elapsed,
        );
      });
    } catch (e) {
      stopwatch.stop();
      setState(() {
        // Replace the "Thinking..." message with the error
        _messages.last = ChatMessage(
          text: "Error: $e", 
          isUser: false,
          responseTime: stopwatch.elapsed,
        );
      });
    } finally {
      setState(() {
        _isGenerating = false;
      });
      _scrollToBottom();
    }
  }

  void _resetConversation() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Reset Conversation'),
          content: const Text('This will clear the chat history and reset the conversation context. Continue?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                setState(() {
                  _messages.clear();
                  _messages.add(ChatMessage(
                    text: "Conversation reset! Ask me anything.",
                    isUser: false,
                  ));
                });
                _llamaService.resetConversation();
                _scrollToBottom();
              },
              child: const Text('Reset'),
            ),
          ],
        );
      },
    );
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _llamaService.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma Chat'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_llamaService.isInitialized && !_isLoading)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _resetConversation,
              tooltip: 'Reset Conversation',
            ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/models'),
            tooltip: 'Model Management',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return _MessageBubble(message: _messages[index]);
              },
            ),
          ),
          if (_isGenerating)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 16),
                  Text("Generating response..."),
                ],
              ),
            ),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              border: Border(
                top: BorderSide(
                  color: Theme.of(context).colorScheme.outline,
                  width: 0.5,
                ),
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                    maxLines: null,
                  ),
                ),
                const SizedBox(width: 8),
                FloatingActionButton(
                  onPressed: (_isLoading ||
                          _isGenerating ||
                          !_llamaService.isInitialized)
                      ? null
                      : _sendMessage,
                  mini: true,
                  child: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final ChatMessage message;

  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color: message.isUser
              ? Theme.of(context).colorScheme.primary
              : Theme.of(context).colorScheme.secondaryContainer,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            SelectableText(
              message.text,
              style: TextStyle(
                color: message.isUser
                    ? Theme.of(context).colorScheme.onPrimary
                    : Theme.of(context).colorScheme.onSecondaryContainer,
              ),
            ),
            if (!message.isUser && message.responseTime != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  'Response time: ${_formatDuration(message.responseTime!)}',
                  style: TextStyle(
                    fontSize: 11,
                    color: (message.isUser
                            ? Theme.of(context).colorScheme.onPrimary
                            : Theme.of(context).colorScheme.onSecondaryContainer)
                        .withOpacity(0.7),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  String _formatDuration(Duration duration) {
    if (duration.inSeconds >= 60) {
      final minutes = duration.inMinutes;
      final seconds = duration.inSeconds % 60;
      return '${minutes}m ${seconds}s';
    } else if (duration.inSeconds >= 1) {
      return '${duration.inSeconds}.${(duration.inMilliseconds % 1000) ~/ 100}s';
    } else {
      return '${duration.inMilliseconds}ms';
    }
  }
}
