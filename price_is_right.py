import logging
import os
import queue
import threading
import time
import gradio as gr
from deal_agent_framework import DealAgentFramework
from log_utils import reformat
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)
POLL_INTERVAL_SECONDS = int(os.getenv("PLANNER_INTERVAL_SECONDS", "120"))


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]

            def update_output(log_data, log_queue, result_queue):
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                finished = False
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            success, payload = result_queue.get_nowait()
                            if success:
                                final_result = payload
                            else:
                                # Keep UI responsive and show the failure in the log panel.
                                log_data.append(
                                    reformat(
                                        f"[App] Run failed: {payload}"
                                    )
                                )
                                final_result = initial_result
                            finished = True
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            if finished:
                                break
                            time.sleep(0.1)

            def get_initial_plot():
                fig = go.Figure()
                fig.update_layout(
                    title="Loading vector DB...",
                    height=400,
                )
                return fig

            def get_plot():
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)
                if len(vectors) == 0:
                    fig = go.Figure()
                    fig.update_layout(
                        title="Vector DB is empty. Add products to see the embedding plot.",
                        height=400,
                        margin=dict(r=5, b=1, l=5, t=30),
                    )
                    return fig

                # Create the 3D scatter plot
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7),
                        )
                    ]
                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),  # Make x-axis twice as long
                        camera=dict(
                            eye=dict(x=1.6, y=1.6, z=0.8)  # Adjust camera position
                        ),
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2),
                )

                return fig

            def do_run():
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                handler = setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                        result_queue.put((True, result))
                    except Exception as exc:
                        logging.exception("Background run failed")
                        result_queue.put((False, str(exc)))

                thread = threading.Thread(target=worker)
                thread.start()

                try:
                    for log_data, output, final_result in update_output(
                        initial_log_data, log_queue, result_queue
                    ):
                        yield log_data, output, final_result
                finally:
                    logger = logging.getLogger()
                    logger.removeHandler(handler)
                    handler.close()

            def do_select(selected_index: gr.SelectData):
                framework = self.get_agent_framework()
                framework.init_agents_as_needed()
                opportunities = framework.memory
                if not opportunities:
                    return
                row = selected_index.index[0]
                if row >= len(opportunities):
                    return
                opportunity = opportunities[row]
                framework.planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px"><strong>The Price is Right</strong> - Autonomous Agent Framework that hunts for deals</div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">A proprietary fine-tuned LLM deployed on Modal and a RAG pipeline with a frontier model collaborate to send push notifications with great online deals.</div>'
                )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    column_count=5,
                    max_height=400,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    gr.Plot(value=get_plot(), show_label=False)

            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            timer = gr.Timer(value=POLL_INTERVAL_SECONDS, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            opportunities_dataframe.select(do_select)

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
