<!-- The exported code uses Tailwind CSS. Install Tailwind CSS in your dev environment to ensure all styles work. -->
<template>
<div id="app" class="min-h-screen bg-gray-900 text-white">
<!-- Header -->
<header class="fixed top-0 left-0 right-0 z-50 bg-gray-900/80 backdrop-blur-sm border-b border-gray-700">
<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
<div class="flex justify-between items-center h-16">
<div class="flex items-center space-x-2">
<span class="text-2xl">🏢</span>
<span class="text-xl font-bold text-blue-400">AI 투자 평가 시스템</span>
</div>
<nav class="hidden md:flex space-x-8">
<a href="#" class="text-gray-300 hover:text-white cursor-pointer transition-colors">대시보드</a>
<a href="#" class="text-gray-300 hover:text-white cursor-pointer transition-colors">분석 기록</a>
<a href="#" class="text-gray-300 hover:text-white cursor-pointer transition-colors">설정</a>
</nav>
<button @click="toggleSidebar" class="md:hidden text-gray-300 hover:text-white cursor-pointer">
<i class="fas fa-bars text-xl"></i>
</button>
</div>
</div>
</header>
<!-- Hero Section -->
<section class="pt-20 pb-16 bg-gradient-to-br from-blue-900 via-purple-900 to-green-900">
<div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
<h1 class="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-green-400 bg-clip-text text-transparent">
AI 기반 스타트업 투자 평가
</h1>
<p class="text-xl md:text-2xl text-gray-300 mb-12">
인공지능이 분석하는 정확하고 객관적인 투자 의사결정 지원 시스템
</p>
<div class="max-w-2xl mx-auto">
<div class="flex flex-col sm:flex-row gap-4">
<input
v-model="startupName"
type="text"
placeholder="분석할 스타트업명을 입력하세요"
class="flex-1 px-6 py-4 text-lg bg-gray-800/50 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-400"
/>
<button
@click="startAnalysis"
:disabled="isAnalyzing || !startupName.trim()"
class="!rounded-button whitespace-nowrap px-8 py-4 bg-gradient-to-r from-blue-600 to-green-600 hover:from-blue-700 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-lg transition-all duration-300 cursor-pointer"
>
<i class="fas fa-search mr-2"></i>
{{ isAnalyzing ? '분석 중...' : '분석 시작' }}
</button>
</div>
</div>
</div>
</section>
<!-- Progress Section -->
<section v-if="showProgress" class="py-8 bg-gray-800">
<div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
<div class="text-center mb-6">
<h3 class="text-xl font-semibold text-blue-400 mb-2">{{ currentStep }}</h3>
<p class="text-gray-400">{{ stepDescription }}</p>
</div>
<div class="w-full bg-gray-700 rounded-full h-3">
<div
class="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-500"
:style="{ width: progress + '%' }"
></div>
</div>
<div class="text-center mt-2">
<span class="text-sm text-gray-400">{{ Math.round(progress) }}% 완료</span>
</div>
</div>
</section>
<!-- Results Dashboard -->
<section v-if="showResults" class="py-16 bg-gray-900">
<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
<!-- Top Results -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
<!-- Investment Decision -->
<div class="bg-gray-800 rounded-xl p-8 text-center">
<h3 class="text-2xl font-bold mb-4">투자 결정</h3>
<div class="inline-flex items-center px-6 py-3 rounded-full text-xl font-bold"
:class="results.decision === '투자' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'">
<i :class="results.decision === '투자' ? 'fas fa-check-circle' : 'fas fa-times-circle'" class="mr-2"></i>
{{ results.decision }}
</div>
</div>
<!-- Overall Score -->
<div class="bg-gray-800 rounded-xl p-8 text-center">
<h3 class="text-2xl font-bold mb-4">종합 점수</h3>
<div class="relative w-32 h-32 mx-auto">
<canvas ref="scoreGauge" width="128" height="128"></canvas>
<div class="absolute inset-0 flex items-center justify-center">
<span class="text-3xl font-bold text-blue-400">{{ animatedScore }}</span>
</div>
</div>
</div>
</div>
<!-- Radar Chart -->
<div class="bg-gray-800 rounded-xl p-8 mb-12 relative group">
<h3 class="text-2xl font-bold text-center mb-8">평가 항목별 분석</h3>
<div class="max-w-2xl mx-auto relative">
<canvas ref="radarChart" width="400" height="400" @mousemove="handleChartHover"></canvas>
<!-- Tooltip -->
<div v-if="tooltipData"
class="absolute bg-gray-900 text-white p-4 rounded-lg shadow-lg z-10 transition-opacity duration-200"
:style="{ left: tooltipPosition.x + 'px', top: tooltipPosition.y + 'px' }">
<h4 class="font-bold mb-2">{{ tooltipData.label }}</h4>
<p class="text-blue-400 text-lg">{{ tooltipData.score }}점</p>
</div>
</div>
</div>
<!-- Detailed Scores -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mb-12">
<div
v-for="(item, index) in scoreItems"
:key="index"
class="bg-gray-800 rounded-xl p-6 text-center hover:scale-105 transition-transform duration-300 cursor-pointer"
>
<div class="text-3xl mb-3">{{ item.icon }}</div>
<h4 class="text-lg font-semibold mb-2">{{ item.name }}</h4>
<div class="text-2xl font-bold text-blue-400 mb-2">{{ item.animatedScore }}점</div>
<p class="text-sm text-gray-400">{{ item.description }}</p>
</div>
</div>
<!-- Download Button -->
<div class="text-center">
<button
@click="downloadPDF"
class="!rounded-button whitespace-nowrap px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold text-lg transition-all duration-300 cursor-pointer"
>
<i class="fas fa-download mr-2"></i>
분석 보고서 PDF 다운로드
</button>
</div>
</div>
</section>
<!-- Sidebar -->
<div
v-if="showSidebar"
class="fixed inset-0 z-50 lg:inset-y-0 lg:right-0 lg:left-auto lg:w-96"
@click.self="closeSidebar"
>
<div class="bg-gray-800 h-full p-6 overflow-y-auto">
<div class="flex justify-between items-center mb-6">
<h3 class="text-xl font-bold">실시간 분석 로그</h3>
<button @click="closeSidebar" class="text-gray-400 hover:text-white cursor-pointer">
<i class="fas fa-times text-xl"></i>
</button>
</div>
<!-- Analysis Log -->
<div class="mb-8">
<h4 class="text-lg font-semibold mb-4 text-blue-400">분석 진행 상황</h4>
<div class="space-y-3">
<div
v-for="(log, index) in analysisLogs"
:key="index"
class="flex items-start space-x-3 p-3 bg-gray-700 rounded-lg"
>
<div class="text-green-400 mt-1">
<i class="fas fa-check-circle"></i>
</div>
<div>
<p class="text-sm font-medium">{{ log.title }}</p>
<p class="text-xs text-gray-400">{{ log.time }}</p>
</div>
</div>
</div>
</div>
<!-- Document Info -->
<div>
<h4 class="text-lg font-semibold mb-4 text-green-400">검색된 문서 정보</h4>
<div class="space-y-3">
<div
v-for="(doc, index) in documents"
:key="index"
class="p-3 bg-gray-700 rounded-lg"
>
<h5 class="text-sm font-medium mb-1">{{ doc.title }}</h5>
<p class="text-xs text-gray-400 mb-2">{{ doc.source }}</p>
<div class="flex justify-between text-xs">
<span class="text-blue-400">신뢰도: {{ doc.reliability }}%</span>
<span class="text-gray-500">{{ doc.date }}</span>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</template>
<script>
export default {
data() {
return {
startupName: '',
isAnalyzing: false,
showProgress: false,
showResults: false,
showSidebar: false,
tooltipData: null,
tooltipPosition: { x: 0, y: 0 },
theme: 'dark',
showErrorState: false,
errorMessage: '',
progress: 0,
currentStep: '',
stepDescription: '',
isDarkMode: true,
showToast: false,
toastMessage: '',
toastType: 'success',
showModal: false,
modalContent: null,
analysisHistory: [],
showSkeleton: false,
currentTaskId: null,
animatedScore: 0,
results: {
decision: '',
overallScore: 0,
scores: {
product: 0,
technology: 0,
growth: 0,
market: 0,
competition: 0
}
},
scoreItems: [
{
name: '상품',
icon: '📱',
score: 0,
animatedScore: 0,
description: '제품 경쟁력 및 차별화'
},
{
name: '기술',
icon: '🔬',
score: 0,
animatedScore: 0,
description: '기술적 우위 및 혁신성'
},
{
name: '성장',
icon: '📈',
score: 0,
animatedScore: 0,
description: '성장 가능성 및 확장성'
},
{
name: '시장',
icon: '🌍',
score: 0,
animatedScore: 0,
description: '시장 규모 및 기회'
},
{
name: '경쟁',
icon: '⚔️',
score: 0,
animatedScore: 0,
description: '경쟁 우위 및 진입장벽'
}
],
analysisLogs: [],
documents: [],
steps: [
{ name: '상품 분석', description: '제품과 서비스를 분석하고 있습니다.' },
{ name: '기술 분석', description: '기술 수준과 차별성을 분석하고 있습니다.' },
{ name: '성장성 분석', description: '성장 가능성을 평가하고 있습니다.' },
{ name: '내부 판단', description: '내부 항목 기준을 확인하고 있습니다.' },
{ name: '시장 분석', description: '시장 환경을 분석하고 있습니다.' },
{ name: '경쟁사 분석', description: '경쟁 현황을 분석하고 있습니다.' },
{ name: '최종 판단', description: '종합 평가를 진행하고 있습니다.' },
{ name: '보고서 생성', description: '분석 보고서를 생성하고 있습니다.' }
],
currentStepIndex: 0
};
},
methods: {
handleChartHover(event) {
const canvas = this.$refs.radarChart;
const rect = canvas.getBoundingClientRect();
const x = event.clientX - rect.left;
const y = event.clientY - rect.top;
const centerX = canvas.width / 2;
const centerY = canvas.height / 2;
// Calculate angle and distance from center
const angle = Math.atan2(y - centerY, x - centerX);
const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
// Define chart areas
const labels = ['상품', '기술', '성장', '시장', '경쟁'];
const scores = [
this.results.scores.product,
this.results.scores.technology,
this.results.scores.growth,
this.results.scores.market,
this.results.scores.competition
];
// Calculate which section was hovered
const sectionAngle = (2 * Math.PI) / 5;
let section = Math.floor(((angle + Math.PI / 2 + 2 * Math.PI) % (2 * Math.PI)) / sectionAngle);
if (distance < 150 && distance > 20) {
this.tooltipData = {
label: labels[section],
score: scores[section]
};
this.tooltipPosition = {
x: x + 20,
y: y - 20
};
} else {
this.tooltipData = null;
}
},
toggleTheme() {
this.theme = this.theme === 'dark' ? 'light' : 'dark';
document.documentElement.classList.toggle('dark');
},
showError(message) {
this.showErrorState = true;
this.errorMessage = message;
setTimeout(() => {
this.showErrorState = false;
this.errorMessage = '';
}, 3000);
},
async fetchWithTimeout(url, options = {}, timeout = 10000) {
const controller = new AbortController();
const id = setTimeout(() => controller.abort(), timeout);
try {
const response = await fetch(url, {
...options,
signal: controller.signal
});
clearTimeout(id);
if (!response.ok) {
throw new Error(`HTTP error! status: ${response.status}`);
}
return response;
} catch (error) {
clearTimeout(id);
throw error;
}
},
async startAnalysis() {
if (!this.startupName.trim()) return;
try {
this.isAnalyzing = true;
this.showProgress = true;
this.showResults = false;
this.progress = 0;
this.currentStepIndex = 0;
this.showSkeleton = true;
this.analysisLogs = [];
this.showToast = false;
this.currentTaskId = null;
// Start evaluation request
try {
const response = await this.fetchWithTimeout('/api/evaluate', {
method: 'POST',
headers: {
'Content-Type': 'application/json'
},
body: JSON.stringify({
startup_name: this.startupName
})
});
const data = await response.json();
this.currentTaskId = data.task_id;
this.showToastMessage('분석이 시작되었습니다.', 'success');
} catch (error) {
console.error('Failed to start evaluation:', error);
this.showToastMessage('분석 시작 중 오류가 발생했습니다.', 'error');
this.isAnalyzing = false;
this.showProgress = false;
this.showSkeleton = false;
return;
}
// Poll status
this.pollStatus();
} catch (error) {
console.error('Analysis failed:', error);
this.showToastMessage('분석 중 오류가 발생했습니다.', 'error');
this.isAnalyzing = false;
this.showProgress = false;
this.showSkeleton = false;
}
},
async pollStatus(retryCount = 0) {
    if (!this.currentTaskId) return;
    if (!this.isAnalyzing) return; // 분석이 중단되면 폴링도 중단
    
    // 재시도 횟수 제한 (최대 5회)
    if (retryCount >= 5) {
        console.error('최대 재시도 횟수 초과');
        this.showToastMessage('연결 오류가 발생했습니다. 페이지를 새로고침해주세요.', 'error');
        this.isAnalyzing = false;
        this.showProgress = false;
        return;
    }
    
    try {
        const response = await this.fetchWithTimeout(`/api/evaluate/${this.currentTaskId}/status`);
        const status = await response.json();
        // console.log('상태 업데이트 받음:', status); // 디버깅용 (주석처리)
        this.progress = status.progress;
        this.currentStep = status.current_step;
        this.stepDescription = status.description;
        // Add to analysis logs
        if (status.current_step && status.current_step !== this.analysisLogs[0]?.title) {
            this.analysisLogs.unshift({
                title: status.current_step,
                time: '방금 전'
            });
        }
        if (status.status === 'completed') {
            await this.fetchResults(this.currentTaskId);
            return;
        }
        if (status.status === 'failed') {
            throw new Error(status.error || '분석 중 오류가 발생했습니다.');
        }
        // Continue polling
        setTimeout(() => this.pollStatus(0), 2000);
    } catch (error) {
        console.error('Status polling failed:', error);
        if (error.name === 'AbortError' || error.message.includes('Failed to fetch')) {
            console.log(`Request was aborted, retrying... (${retryCount + 1}/5)`);
            // AbortError인 경우 재시도
            setTimeout(() => this.pollStatus(retryCount + 1), 1000);
return;
}
if (error.message.includes('Failed to fetch')) {
console.log('Network error, retrying...');
// 네트워크 에러인 경우 재시도
setTimeout(() => this.pollStatus(), 2000);
return;
}
this.showToastMessage('분석 상태 확인 중 오류가 발생했습니다.', 'error');
this.isAnalyzing = false;
this.showProgress = false;
this.showSkeleton = false;
}
},
async fetchResults(taskId) {
try {
const response = await this.fetchWithTimeout(`/api/evaluate/${taskId}/result`);
const results = await response.json();
// Update results with API response
this.results = {
decision: results.decision,
overallScore: results.overall_score,
scores: {
product: results.scores.product,
technology: results.scores.technology,
growth: results.scores.growth,
market: results.scores.market,
competition: results.scores.competition
}
};
// Update score items
this.scoreItems.forEach(item => {
switch(item.name) {
case '상품':
item.score = this.results.scores.product;
break;
case '기술':
item.score = this.results.scores.technology;
break;
case '성장':
item.score = this.results.scores.growth;
break;
case '시장':
item.score = this.results.scores.market;
break;
case '경쟁':
item.score = this.results.scores.competition;
break;
}
});
// Save to analysis history
const analysisRecord = {
id: taskId,
name: this.startupName,
date: new Date().toISOString(),
score: this.results.overallScore,
decision: this.results.decision
};
this.analysisHistory.push(analysisRecord);
localStorage.setItem('analysisHistory', JSON.stringify(this.analysisHistory));
this.showSkeleton = false;
this.isAnalyzing = false; // 폴링 중단을 위해 추가
this.showProgress = false; // 진행률 숨기기
this.completeAnalysis();
this.showToastMessage('분석이 완료되었습니다.', 'success');
} catch (error) {
console.error('Failed to fetch results:', error);
this.showToastMessage('결과 조회 중 오류가 발생했습니다.', 'error');
this.isAnalyzing = false; // 에러 시에도 폴링 중단
this.showProgress = false;
}
},

showToastMessage(message, type = 'success') {
this.toastMessage = message;
this.toastType = type;
this.showToast = true;
setTimeout(() => {
this.showToast = false;
}, 3000);
},
toggleDarkMode() {
this.isDarkMode = !this.isDarkMode;
document.documentElement.classList.toggle('dark');
},
showDetailModal(item) {
this.modalContent = item;
this.showModal = true;
},
mounted() {
// Load analysis history from localStorage
const savedHistory = localStorage.getItem('analysisHistory');
if (savedHistory) {
this.analysisHistory = JSON.parse(savedHistory);
}
// Add keyboard navigation
document.addEventListener('keydown', (e) => {
if (e.key === 'Escape' && this.showModal) {
this.showModal = false;
}
});
},

completeAnalysis() {
this.isAnalyzing = false;
this.showProgress = false;
this.showResults = true;
// Animate scores
this.animateScores();
// Draw charts after DOM update
this.$nextTick(() => {
this.drawScoreGauge();
this.drawRadarChart();
});
},
animateScores() {
// Animate overall score
this.animateValue(0, this.results.overallScore, 2000, (value) => {
this.animatedScore = Math.round(value);
});
// Animate individual scores
this.scoreItems.forEach((item, index) => {
setTimeout(() => {
this.animateValue(0, item.score, 1500, (value) => {
item.animatedScore = Math.round(value);
});
}, index * 200);
});
},
animateValue(start, end, duration, callback) {
const startTime = performance.now();
const animate = (currentTime) => {
const elapsed = currentTime - startTime;
const progress = Math.min(elapsed / duration, 1);
const value = start + (end - start) * this.easeOutCubic(progress);
callback(value);
if (progress < 1) {
requestAnimationFrame(animate);
}
};
requestAnimationFrame(animate);
},
easeOutCubic(t) {
return 1 - Math.pow(1 - t, 3);
},
drawScoreGauge() {
const canvas = this.$refs.scoreGauge;
if (!canvas) return;
const ctx = canvas.getContext('2d');
const centerX = canvas.width / 2;
const centerY = canvas.height / 2;
const radius = 50;
// Clear canvas
ctx.clearRect(0, 0, canvas.width, canvas.height);
// Draw background circle
ctx.beginPath();
ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
ctx.strokeStyle = '#374151';
ctx.lineWidth = 8;
ctx.stroke();
// Draw score arc
const scoreAngle = (this.results.overallScore / 100) * 2 * Math.PI;
ctx.beginPath();
ctx.arc(centerX, centerY, radius, -Math.PI / 2, -Math.PI / 2 + scoreAngle);
ctx.strokeStyle = this.results.overallScore >= 70 ? '#10B981' : '#EF4444';
ctx.lineWidth = 8;
ctx.lineCap = 'round';
ctx.stroke();
},
drawRadarChart() {
const canvas = this.$refs.radarChart;
if (!canvas) return;
const ctx = canvas.getContext('2d');
const centerX = canvas.width / 2;
const centerY = canvas.height / 2;
const radius = 150;

// Animation progress
let progress = 0;
const animationDuration = 2000; // 2 seconds
const startTime = performance.now();

const labels = ['상품', '기술', '성장', '시장', '경쟁'];
const scores = [
this.results.scores.product,
this.results.scores.technology,
this.results.scores.growth,
this.results.scores.market,
this.results.scores.competition
];

const colors = [
'rgba(59, 130, 246, 0.7)',   // Blue
'rgba(16, 185, 129, 0.7)',   // Green
'rgba(245, 158, 11, 0.7)',   // Yellow
'rgba(236, 72, 153, 0.7)',   // Pink
'rgba(139, 92, 246, 0.7)'    // Purple
];

const animate = (currentTime) => {
const elapsed = currentTime - startTime;
progress = Math.min(elapsed / animationDuration, 1);

// Clear canvas
ctx.clearRect(0, 0, canvas.width, canvas.height);
// Draw scale labels and grid
for (let i = 1; i <= 5; i++) {
ctx.beginPath();
const gridRadius = (radius / 5) * i;
for (let j = 0; j < 5; j++) {
const angle = (j * 2 * Math.PI) / 5 - Math.PI / 2;
const x = centerX + Math.cos(angle) * gridRadius;
const y = centerY + Math.sin(angle) * gridRadius;
if (j === 0) {
ctx.moveTo(x, y);
} else {
ctx.lineTo(x, y);
}
}
ctx.closePath();
ctx.strokeStyle = '#374151';
ctx.lineWidth = 1;
ctx.stroke();

// Draw scale numbers
const scaleValue = (i * 20).toString();
ctx.fillStyle = '#9CA3AF';
ctx.font = '12px sans-serif';
ctx.fillText(scaleValue, centerX + 5, centerY - gridRadius);
}
// Draw axes
for (let i = 0; i < 5; i++) {
ctx.beginPath();
const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
ctx.moveTo(centerX, centerY);
ctx.lineTo(
centerX + Math.cos(angle) * radius,
centerY + Math.sin(angle) * radius
);
ctx.strokeStyle = '#374151';
ctx.lineWidth = 1;
ctx.stroke();
}
// Draw data for each category
for (let i = 0; i < 5; i++) {
ctx.beginPath();
const startAngle = ((i * 2 * Math.PI) / 5) - Math.PI / 2;
const endAngle = (((i + 1) * 2 * Math.PI) / 5) - Math.PI / 2;
const value = (scores[i] / 100) * radius * progress;

// Draw sector
ctx.moveTo(centerX, centerY);
ctx.lineTo(
centerX + Math.cos(startAngle) * value,
centerY + Math.sin(startAngle) * value
);
ctx.arc(
centerX,
centerY,
value,
startAngle,
endAngle
);
ctx.lineTo(centerX, centerY);
ctx.closePath();

// Fill with category color
ctx.fillStyle = colors[i];
ctx.fill();
ctx.strokeStyle = colors[i].replace('0.7', '1');
ctx.lineWidth = 2;
ctx.stroke();

// Draw point
const pointX = centerX + Math.cos((startAngle + endAngle) / 2) * value;
const pointY = centerY + Math.sin((startAngle + endAngle) / 2) * value;
ctx.beginPath();
ctx.arc(pointX, pointY, 6, 0, 2 * Math.PI);
ctx.fillStyle = '#FFFFFF';
ctx.fill();
ctx.strokeStyle = colors[i].replace('0.7', '1');
ctx.lineWidth = 2;
ctx.stroke();
}
// Draw points
for (let i = 0; i < 5; i++) {
const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
const value = (scores[i] / 100) * radius;
const x = centerX + Math.cos(angle) * value;
const y = centerY + Math.sin(angle) * value;
ctx.beginPath();
ctx.arc(x, y, 4, 0, 2 * Math.PI);
ctx.fillStyle = '#3B82F6';
ctx.fill();
}
// Draw labels with score
ctx.textAlign = 'center';
for (let i = 0; i < 5; i++) {
const angle = ((i * 2 * Math.PI) / 5) - Math.PI / 2;
const labelRadius = radius + 30;
const x = centerX + Math.cos(angle) * labelRadius;
const y = centerY + Math.sin(angle) * labelRadius;

// Label text
ctx.font = 'bold 14px sans-serif';
ctx.fillStyle = colors[i].replace('0.7', '1');
ctx.fillText(labels[i], x, y);

// Score text
ctx.font = '12px sans-serif';
ctx.fillStyle = '#D1D5DB';
ctx.fillText(`${Math.round(scores[i] * progress)}점`, x, y + 20);
}

if (progress < 1) {
requestAnimationFrame(animate);
}
};

requestAnimationFrame(animate);
},
toggleSidebar() {
this.showSidebar = !this.showSidebar;
},
closeSidebar() {
this.showSidebar = false;
},
async downloadPDF() {
try {
if (!this.currentTaskId) {
this.showToastMessage('분석 기록을 찾을 수 없습니다.', 'error');
return;
}
const response = await this.fetchWithTimeout(`/api/evaluate/${this.currentTaskId}/pdf`, {
method: 'GET'
});
const blob = await response.blob();
const url = window.URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url;
link.download = `${this.startupName}_투자평가보고서.pdf`;
document.body.appendChild(link);
link.click();
document.body.removeChild(link);
window.URL.revokeObjectURL(url);
this.showToastMessage('PDF 다운로드가 시작되었습니다.', 'success');
} catch (error) {
console.error('PDF download failed:', error);
this.showToastMessage('PDF 다운로드 중 오류가 발생했습니다.', 'error');
}
}
}
};
</script>
<style scoped>
.\!rounded-button {
border-radius: 0.5rem !important;
}
@media (max-width: 768px) {
.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3.xl\\:grid-cols-5 {
grid-template-columns: repeat(1, minmax(0, 1fr));
}
}
@media (min-width: 768px) {
.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3.xl\\:grid-cols-5 {
grid-template-columns: repeat(2, minmax(0, 1fr));
}
}
@media (min-width: 1024px) {
.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3.xl\\:grid-cols-5 {
grid-template-columns: repeat(3, minmax(0, 1fr));
}
}
@media (min-width: 1280px) {
.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3.xl\\:grid-cols-5 {
grid-template-columns: repeat(5, minmax(0, 1fr));
}
}
</style>