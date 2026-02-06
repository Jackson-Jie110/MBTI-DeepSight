document.addEventListener('DOMContentLoaded', () => {
  const MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP',
  ];

  const modal = document.getElementById('feedback-modal');
  const card = document.getElementById('feedback-card');
  const btn = document.getElementById('feedback-btn');
  const closeBtn = document.getElementById('close-feedback');
  const submitBtn = document.getElementById('submit-feedback-btn');
  const stars = document.querySelectorAll('.star-btn');
  const contentInput = document.getElementById('feedback-content');
  const mbtiSelect = document.getElementById('feedback-mbti-select');

  if (!modal || !card || !closeBtn || !submitBtn || stars.length === 0 || !mbtiSelect) {
    return;
  }

  let currentRating = 0;

  function detectMBTI() {
    const typeEl = document.querySelector('.result-hero__typecode');
    if (typeEl) {
      const text = typeEl.innerText.trim().toUpperCase();
      if (MBTI_TYPES.includes(text)) {
        return text;
      }
    }

    const urlParams = new URLSearchParams(window.location.search);
    const urlType = urlParams.get('type');
    if (urlType && MBTI_TYPES.includes(urlType.toUpperCase())) {
      return urlType.toUpperCase();
    }

    const candidates = document.querySelectorAll('span, h1, h2, div');
    for (const element of candidates) {
      const txt = element.innerText.trim().toUpperCase();
      if (MBTI_TYPES.includes(txt) && element.innerText.length < 10) {
        return txt;
      }
    }

    return null;
  }

  let attempts = 0;
  const maxAttempts = 20;
  const poller = setInterval(() => {
    const type = detectMBTI();
    if (type) {
      console.log('✅ MBTI Detected via Polling:', type);
      localStorage.setItem('user_mbti_cache', type);
      clearInterval(poller);
    } else {
      attempts += 1;
      if (attempts >= maxAttempts) {
        clearInterval(poller);
      }
    }
  }, 500);

  function hideModal() {
    modal.classList.add('opacity-0');
    card.classList.remove('scale-100');
    card.classList.add('scale-95');
    setTimeout(() => {
      modal.classList.add('hidden');
    }, 300);
  }

  function updateStars(value) {
    const intValue = Number(value);
    stars.forEach((star) => {
      const starVal = Number(star.getAttribute('data-val'));
      if (starVal <= intValue) {
        star.classList.remove('text-gray-300');
        star.classList.add('text-yellow-400');
      } else {
        star.classList.add('text-gray-300');
        star.classList.remove('text-yellow-400');
      }
    });
  }

  function resetButtonState() {
    submitBtn.innerText = '提交反馈';
    submitBtn.disabled = false;
  }

  window.openFeedbackModal = function openFeedbackModal() {
    let finalType = localStorage.getItem('user_mbti_cache');
    if (!finalType) {
      finalType = detectMBTI();
    }

    if (finalType) {
      mbtiSelect.value = finalType;
    } else {
      mbtiSelect.value = 'Unknown';
    }

    modal.classList.remove('hidden');
    setTimeout(() => {
      modal.classList.remove('opacity-0');
      card.classList.remove('scale-95');
      card.classList.add('scale-100');
    }, 10);
  };

  if (btn) {
    btn.addEventListener('click', window.openFeedbackModal);
  }
  closeBtn.addEventListener('click', hideModal);

  stars.forEach((star) => {
    star.addEventListener('click', function onClick() {
      currentRating = Number(this.getAttribute('data-val'));
      updateStars(currentRating);
    });
  });

  submitBtn.addEventListener('click', async () => {
    if (currentRating === 0) {
      alert('请先点亮星星打分哦~');
      return;
    }

    const content = contentInput ? contentInput.value : '';
    const mbtiType = mbtiSelect.value;

    submitBtn.innerText = '提交中...';
    submitBtn.disabled = true;

    try {
      const response = await fetch('/submit_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rating: currentRating,
          content,
          mbti_type: mbtiType,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      submitBtn.innerText = '✅ 收到，感谢！';
      setTimeout(() => {
        hideModal();
        resetButtonState();
        if (contentInput) {
          contentInput.value = '';
        }
        currentRating = 0;
        updateStars(0);
      }, 1500);
    } catch (error) {
      console.error(error);
      alert('提交失败，请稍后重试');
      resetButtonState();
    }
  });
});
