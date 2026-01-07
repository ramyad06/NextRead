const API_BASE_URL = 'http://localhost:8000';

const userIdInput = document.getElementById('user-id-input');
const getRecsBtn = document.getElementById('get-recs-btn');
const surpriseMeBtn = document.getElementById('surprise-me-btn');
const resultsContainer = document.getElementById('results-container');
const cardsWrapper = document.getElementById('cards-wrapper');
const strategyBadge = document.getElementById('strategy-badge');
const errorMessage = document.getElementById('error-message');
const spinner = document.querySelector('.spinner');
const btnText = document.querySelector('.primary-btn span');

// Event Listeners
getRecsBtn.addEventListener('click', () => {
    const userId = userIdInput.value;
    if (!userId && userId !== '0') {
        alert('Please enter a User ID');
        return;
    }
    fetchRecommendations(userId);
});

surpriseMeBtn.addEventListener('click', () => {
    // Generate a random large ID to ensure cold start
    const randomId = Math.floor(Math.random() * 1000) + 10000;
    userIdInput.value = randomId;
    fetchRecommendations(randomId);
});

userIdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const userId = userIdInput.value;
        if (userId) fetchRecommendations(userId);
    }
});

async function fetchRecommendations(userId) {
    setLoading(true);
    resetUI();

    try {
        const response = await fetch(`${API_BASE_URL}/recommend/${userId}?top_k=5`);

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error fetching recommendations:', error);
        showError();
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    // Update Badge based on strategy
    strategyBadge.textContent = data.strategy === 'popularity'
        ? 'Popularity Fallback'
        : 'Hybrid Personalized';

    strategyBadge.className = data.strategy === 'popularity'
        ? 'badge popularity'
        : 'badge';

    // Clear previous results
    cardsWrapper.innerHTML = '';

    if (!data.recommendations || data.recommendations.length === 0) {
        cardsWrapper.innerHTML = '<p class="text-secondary">No recommendations found.</p>';
        return;
    }

    // Create Cards with staggered animation delay
    data.recommendations.forEach((rec, index) => {
        const card = document.createElement('a');
        card.href = rec.url;
        card.target = '_blank';
        card.className = 'card';
        card.style.animation = `fadeInUp 0.5s ${index * 0.1}s backwards`;

        card.innerHTML = `
            <div class="card-content">
                <div class="card-title">${rec.title}</div>
                <div class="card-meta">
                    <span class="score-pill">Score: ${rec.score.toFixed(3)}</span>
                    <span class="url-hint">${new URL(rec.url).hostname}</span>
                </div>
            </div>
            <div class="card-arrow">→</div>
        `;

        cardsWrapper.appendChild(card);
    });

    resultsContainer.classList.remove('hidden');
}

function setLoading(isLoading) {
    if (isLoading) {
        spinner.classList.remove('hidden');
        btnText.style.opacity = '0';
        getRecsBtn.disabled = true;
    } else {
        spinner.classList.add('hidden');
        btnText.style.opacity = '1';
        getRecsBtn.disabled = false;
    }
}

function resetUI() {
    resultsContainer.classList.add('hidden');
    errorMessage.classList.add('hidden');
}

function showError() {
    errorMessage.classList.remove('hidden');
}
