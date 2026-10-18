import '@fontsource-variable/martian-mono/wdth.css';
import '@fontsource-variable/atkinson-hyperlegible-next/wght.css';
import './style.css';

// contact details are stored reversed and base64-encoded to keep them out of the plain HTML
document.querySelectorAll<HTMLElement>('[data-obf]').forEach((el) => {
  const text = atob(el.dataset.obf!).split('').reverse().join('');
  el.textContent = text;
  if (el instanceof HTMLAnchorElement) el.href = `mailto:${text}`;
});

// back to the view the visitor came from (simple or advanced): history keeps its state where the browser can, the
// referrer keeps its settings otherwise; a direct visit goes to the start page
const back = document.querySelector<HTMLAnchorElement>('.back a');
const from = document.referrer && new URL(document.referrer).origin === location.origin ? document.referrer : '';
if (back && from) {
  back.href = from;
  back.onclick = (ev) => {
    if (history.length < 2) return;
    ev.preventDefault();
    history.back();
  };
}
