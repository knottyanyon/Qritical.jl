import { renderMathInHtml } from './math-render.js';

let activeCard = null;

function dismiss() {
  if (activeCard) {
    activeCard.remove();
    activeCard = null;
    document.removeEventListener('click', onOutsideClick, true);
  }
}

function onOutsideClick(e) {
  if (activeCard && !activeCard.contains(e.target)) dismiss();
}

export function attachDetailCard(target, detailHtml) {
  const el = typeof target.node === 'function' ? target.node() : target;
  el.style.cursor = 'pointer';
  el.classList.add('diagram-node--interactive');
  el.addEventListener('click', (e) => {
    e.stopPropagation();
    dismiss();
    const card = document.createElement('div');
    card.className = 'flow-detail-card';
    renderMathInHtml(detailHtml, card);
    card.style.position = 'fixed';
    card.style.left = `${e.clientX + 12}px`;
    card.style.top = `${e.clientY + 12}px`;
    document.body.appendChild(card);
    activeCard = card;
    setTimeout(() => document.addEventListener('click', onOutsideClick, true), 0);
  });
}
