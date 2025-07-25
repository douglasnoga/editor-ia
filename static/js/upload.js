/**
 * Editor IA - Upload and Progress Handling
 * JavaScript for handling file uploads, progress tracking, and user interactions
 */

class VideoUploadManager {
    constructor() {
        this.processingId = null;
        this.statusCheckInterval = null;
        this.currentFile = null;
        
        this.initializeEventListeners();
        this.setupDragAndDrop();
    }

    /**
     * Initialize event listeners for form elements
     */
    initializeEventListeners() {
        // File input change
        const fileInput = document.getElementById('videoFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // Process button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.addEventListener('click', () => this.processVideo());
        }

        // Template selector
        const templateSelect = document.getElementById('videoType');
        if (templateSelect) {
            templateSelect.addEventListener('change', (e) => this.handleTemplateChange(e));
        }
    }

    /**
     * Setup drag and drop functionality
     */
    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.highlight(uploadArea), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.unhighlight(uploadArea), false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    /**
     * Prevent default drag behaviors
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Highlight drop area
     */
    highlight(element) {
        element.classList.add('drag-over');
    }

    /**
     * Remove highlight from drop area
     */
    unhighlight(element) {
        element.classList.remove('drag-over');
    }

    /**
     * Handle file drop
     */
    handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect({ target: { files } });
        }
    }

    /**
     * Handle file selection
     */
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        this.currentFile = file;
        this.updateFileInfo(file);
        this.enableProcessButton();
    }

    /**
     * Validate selected file
     */
    validateFile(file) {
        const validTypes = [
            'video/mp4', 'video/quicktime', 'video/x-msvideo',
            'audio/mpeg', 'audio/wav', 'audio/mp4'
        ];
        
        const validExtensions = ['.mp4', '.mov', '.avi', '.mp3', '.wav', '.m4a'];
        
        // Check file type
        if (!validTypes.includes(file.type)) {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            if (!validExtensions.includes(extension)) {
                this.showError('Formato de arquivo não suportado. Use MP4, MOV, AVI, MP3, WAV ou M4A.');
                return false;
            }
        }

        // Check file size (2GB limit)
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (file.size > maxSize) {
            this.showError('Arquivo muito grande. Tamanho máximo: 2GB.');
            return false;
        }

        return true;
    }

    /**
     * Update file information display
     */
    updateFileInfo(file) {
        const uploadText = document.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = `Arquivo selecionado: ${file.name}`;
        }

        const uploadFormats = document.querySelector('.upload-formats');
        if (uploadFormats) {
            const fileSize = this.formatFileSize(file.size);
            uploadFormats.textContent = `Tamanho: ${fileSize}`;
        }
    }

    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Enable the process button
     */
    enableProcessButton() {
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.disabled = false;
            processButton.textContent = 'PROCESSAR VÍDEO';
        }
    }

    /**
     * Handle template selection change
     */
    handleTemplateChange(e) {
        const template = e.target.value;
        const instructionsTextarea = document.getElementById('instructions');
        
        if (instructionsTextarea) {
            // Update placeholder based on template
            const placeholders = {
                'youtube_cuts': 'Encontre os melhores momentos da live para criar um vídeo de 6-12 minutos',
                'vsl': 'Crie um vídeo de vendas persuasivo de 10-20 minutos com gancho, problema, solução e CTA',
                'social_reels': 'Extraia o momento mais viral e impactante para um reel de 15-90 segundos',
                'educational': 'Organize o conteúdo educacional em uma sequência lógica e clara',
                'advertising': 'Crie um anúncio impactante de 30-120 segundos com call to action forte',
                'general': 'Descreva como você gostaria que o vídeo fosse editado'
            };
            
            instructionsTextarea.placeholder = placeholders[template] || placeholders['general'];
        }
    }

    /**
     * Process video
     */
    async processVideo() {
        if (!this.currentFile) {
            this.showError('Por favor, selecione um arquivo de vídeo');
            return;
        }

        const videoType = document.getElementById('videoType').value;
        const instructions = document.getElementById('instructions').value;
        const provider = document.getElementById('provider').value;

        // Prepare form data
        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('video_type', videoType);
        formData.append('custom_instructions', instructions);
        formData.append('transcription_provider', provider);

        try {
            // Show processing UI
            this.showProcessingUI();
            
            // Start upload
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.processingId = result.processing_id;
                this.updateStatus('Upload concluído, iniciando processamento...', 'processing');
                this.startStatusChecking();
            } else {
                this.showError(result.detail || 'Erro no upload');
            }
        } catch (error) {
            this.showError('Erro de conexão: ' + error.message);
        }
    }

    /**
     * Show processing UI
     */
    showProcessingUI() {
        const statusSection = document.getElementById('statusSection');
        const processButton = document.getElementById('processButton');
        
        if (statusSection) {
            statusSection.classList.add('show');
            statusSection.classList.add('fade-in');
        }
        
        if (processButton) {
            processButton.disabled = true;
            processButton.innerHTML = '<span class="loading-spinner"></span>Processando...';
        }
        
        this.updateStatus('Iniciando upload...', 'processing');
        this.updateProgress(0);
    }

    /**
     * Start status checking
     */
    startStatusChecking() {
        this.statusCheckInterval = setInterval(() => {
            this.checkStatus();
        }, 2000); // Check every 2 seconds
    }

    /**
     * Check processing status
     */
    async checkStatus() {
        if (!this.processingId) return;

        try {
            const response = await fetch(`/api/status/${this.processingId}`);
            const status = await response.json();

            if (response.ok) {
                this.updateStatus(status.current_step, status.status);
                this.updateProgress(status.progress);

                if (status.status === 'completed') {
                    this.handleProcessingComplete();
                } else if (status.status === 'failed') {
                    this.handleProcessingError(status.error);
                }
            } else {
                this.handleProcessingError('Erro ao verificar status');
            }
        } catch (error) {
            this.handleProcessingError('Erro de conexão: ' + error.message);
        }
    }

    /**
     * Update status display
     */
    updateStatus(message, status) {
        const statusText = document.getElementById('statusText');
        const statusIndicator = document.getElementById('statusIndicator');
        
        if (statusText) {
            statusText.textContent = message;
        }
        
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${status}`;
        }
    }

    /**
     * Update progress bar
     */
    updateProgress(progress) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${progress}% concluído`;
        }
    }

    /**
     * Handle processing completion
     */
    handleProcessingComplete() {
        this.stopStatusChecking();
        
        // Show download links
        const downloadSection = document.getElementById('downloadSection');
        if (downloadSection) {
            downloadSection.style.display = 'block';
            downloadSection.classList.add('fade-in');
            
            // Update download links
            const xmlLink = document.getElementById('xmlLink');
            const guideLink = document.getElementById('guideLink');
            
            if (xmlLink) {
                xmlLink.href = `/api/download/${this.processingId}/xml`;
                xmlLink.textContent = 'Baixar XML Premiere';
            }
            
            if (guideLink) {
                guideLink.href = `/api/download/${this.processingId}/guide`;
                guideLink.textContent = 'Baixar Guia de Cortes';
            }
        }
        
        // Reset process button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.disabled = false;
            processButton.textContent = 'PROCESSAR OUTRO VÍDEO';
        }
        
        this.showSuccess('Processamento concluído com sucesso!');
    }

    /**
     * Handle processing error
     */
    handleProcessingError(error) {
        this.stopStatusChecking();
        
        // Reset process button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.disabled = false;
            processButton.textContent = 'TENTAR NOVAMENTE';
        }
        
        this.showError(error);
    }

    /**
     * Stop status checking
     */
    stopStatusChecking() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        this.showMessage(message, 'error');
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    /**
     * Show message
     */
    showMessage(message, type) {
        // Remove existing messages
        const existingMessages = document.querySelectorAll('.error-message, .success-message');
        existingMessages.forEach(msg => msg.remove());
        
        // Create new message
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message fade-in`;
        messageDiv.textContent = message;
        
        // Insert message
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(messageDiv, container.firstChild);
            
            // Auto-hide success messages
            if (type === 'success') {
                setTimeout(() => {
                    messageDiv.remove();
                }, 5000);
            }
        }
    }

    /**
     * Reset form
     */
    resetForm() {
        this.currentFile = null;
        this.processingId = null;
        
        // Reset file input
        const fileInput = document.getElementById('videoFile');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Reset upload area
        const uploadText = document.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = 'Arraste um arquivo aqui ou clique para selecionar';
        }
        
        const uploadFormats = document.querySelector('.upload-formats');
        if (uploadFormats) {
            uploadFormats.textContent = 'Formatos aceitos: MP4, MOV, AVI, MP3, WAV, M4A';
        }
        
        // Reset instructions
        const instructionsTextarea = document.getElementById('instructions');
        if (instructionsTextarea) {
            instructionsTextarea.value = '';
        }
        
        // Hide status section
        const statusSection = document.getElementById('statusSection');
        if (statusSection) {
            statusSection.classList.remove('show');
        }
        
        // Hide download section
        const downloadSection = document.getElementById('downloadSection');
        if (downloadSection) {
            downloadSection.style.display = 'none';
        }
        
        // Reset process button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.disabled = true;
            processButton.textContent = 'SELECIONE UM ARQUIVO';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uploadManager = new VideoUploadManager();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VideoUploadManager;
}