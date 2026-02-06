window.openFeedbackModal = function openFeedbackModal() {
  const modal = document.getElementById('feedback-modal');
  const card = document.getElementById('feedback-card');
  if (!modal || !card) {
    return;
  }

  modal.classList.remove('hidden');
  setTimeout(() => {
    modal.classList.remove('opacity-0');
    card.classList.remove('scale-95');
    card.classList.add('scale-100');
  }, 10);
};

document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('feedback-modal');
  const card = document.getElementById('feedback-card');
  const btn = document.getElementById('feedback-btn');
  const closeBtn = document.getElementById('close-feedback');
  const submitBtn = document.getElementById('submit-feedback-btn');
  const stars = document.querySelectorAll('.star-btn');
  const contentInput = document.getElementById('feedback-content');

  if (!modal || !card || !closeBtn || !submitBtn || stars.length === 0) {
    return;
  }

  let currentRating = 0;

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

    const urlParams = new URLSearchParams(window.location.search);
    let mbtiType = urlParams.get('type');

    if (!mbtiType) {
      console.warn('MBTI type not found in URL');
      mbtiType = 'Unknown';
    }

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
