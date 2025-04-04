import ast
import gym
import json
import time
import requests
from bs4 import BeautifulSoup


# Function for cleaning and normalizing text encoding from Wikipedia responses
def clean_str(p):
    """Clean and normalize text encoding from Wikipedia responses
    Handles common encoding issues in Wikipedia's HTML content through:
    1. Unicode-escape decoding
    2. Latin-1 encoding
    3. UTF-8 final decoding
    """
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        return isinstance(x, str)


# Wikipedia interaction Class
class WikiEnv(gym.Env):
    """Wikipedia interaction environment implementing ReAct-style actions
    Action Space:
    - search[entity]: Search Wikipedia for an entity
    - lookup[keyword]: Lookup keyword in current page
    - finish[answer]: Submit final answer
    - think[content]: Internal reasoning step

    Observation Space:
    - Text responses from Wikipedia and system messages
    """

    def __init__(self):
        """Initialize Wikipedia environment with empty state"""
        super().__init__()
        self.page = None  # Current Wikipedia page content
        self.obs = None  # Current observation text
        self.lookup_keyword = None  # Current search keyword
        self.lookup_list = None  # List of matching paragraphs
        self.lookup_cnt = None  # Current lookup position
        self.steps = 0  # Interaction step counter
        self.answer = None  # Final answer storage
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0  # Total time spent searching
        self.num_searches = 0  # Total search operations count

    def _get_obs(self):
        """Get current observation text"""
        return self.obs

    def _get_info(self):
        """Get environment information"""
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
        """Reset environment to initial state
        Returns:
            observation: Initial prompt text
            info: Dictionary with step count and answer
        """
        # Initialize with interaction instructions
        self.obs = "Interact with Wikipedia using search[], lookup[], and finish[].\n"
        # Clear state variables
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        """Build list of content matching keyword in current page
        Args:
            keyword: Search term to look for
        Returns:
            List of paragraphs containing the keyword
        """
        if self.page is None:
            return []

        # Process page content into paragraphs
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Split paragraphs into individual sentences
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        # Filter sentences containing keyword
        return [p for p in sentences if keyword.lower() in p.lower()]

    @staticmethod
    def get_page_obs(page):
        """Create observation from raw page content
        Args:
            page: Full Wikipedia page text
        Returns:
            Concatenated first 5 sentences from page
        """
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        # Return first 5 sentences as initial observation
        return " ".join(sentences[:5])

    def search_step(self, entity):
        """Execute Wikipedia search operation
        Args:
            entity: Search term to look up on Wikipedia
        """
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"

        # Time the search request
        start_time = time.time()
        response_text = requests.get(search_url).text
        self.search_time += time.time() - start_time
        self.num_searches += 1

        # Parse HTML response
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # Search results page (no exact match)
            self.result_titles = [
                clean_str(div.get_text().strip()) for div in result_divs
            ]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:  # Direct article page
            page_content = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]

            # Handle disambiguation pages
            if any("may refer to:" in p for p in page_content):
                self.search_step("[" + entity + "]")  # Recursive search with brackets
            else:
                self.page = ""
                # Build cleaned page content
                for p in page_content:
                    if len(p.split(" ")) > 2:  # Filter short paragraphs
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                # Generate initial observation from page
                self.obs = self.get_page_obs(self.page)
                # Reset lookup state
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        """Execute environment step
        Args:
            action: String command from agent
        Returns:
            observation: Updated environment text
            reward: Always 0 (handled externally)
            done: Episode completion flag
            info: Step metadata
        """
        reward = 0
        done = False
        action = action.strip()

        # Check for the answer
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()

        # Process the search action
        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search[") : -1]
            self.search_step(entity)

        # Process the lookup action
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup[") : -1]
            # Reset lookup state for new keywords
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            # Handle lookup navigation
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                result = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                result += self.lookup_list[self.lookup_cnt]
                self.obs = result
                self.lookup_cnt += 1

        # Process finish action
        elif action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish[") : -1]
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"

        # Process think action
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."

        # Invalid action handling
        else:
            self.obs = f"Invalid action: {action}"

        self.steps += 1
        return self.obs, reward, done, self._get_info()

    # Function for getting time info
    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }
