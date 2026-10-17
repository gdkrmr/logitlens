// styles load at once, the view itself only when the link is not one for the advanced view
import '@fontsource-variable/martian-mono/wdth.css';
import '@fontsource-variable/atkinson-hyperlegible-next/wght.css';
import './style.css';
import './simple.css';
import { advancedOnly } from './views';

if (advancedOnly(new URLSearchParams(location.search))) location.replace(`advanced.html${location.search}`);
else import('./simple');
