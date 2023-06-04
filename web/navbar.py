import pynecone as pc


def navbar(component):
    return pc.box(
        pc.box(
            pc.hstack(
                pc.hstack(
                    pc.image(src="icon.svg", width="50px"),
                    pc.desktop_only(
                        pc.heading(
                            "Error-Free Image Scraping",
                            font_size="1.5em",
                            white_space="nowrap",
                        ),
                        pc.flex(
                            pc.badge("Beta", color_scheme="red"),
                        ),
                    ),
                ),
                pc.menu(
                    pc.menu_button(
                        pc.icon(
                            tag="hamburger",
                            width="1.5em",
                            height="1.5em",
                            _hover={
                                "cursor": "pointer",
                            },
                        ),
                    ),
                    pc.menu_list(
                        pc.mobile_and_tablet(
                            pc.menu_item(
                                pc.badge("Beta", color_scheme="red"),
                            ),
                            pc.menu_divider(),
                        ),
                        pc.link(pc.menu_item("CLIP"), href="/"),
                        pc.menu_divider(),
                        pc.link(pc.menu_item("MOCO"), href="/moco"),
                        pc.menu_divider(),
                        pc.link(pc.menu_item("ImageBind"), href="/bind"),
                    ),
                ),
                justify="space-between",
                border_bottom="0.2em solid #F0F0F0",
                padding_x="2em",
                padding_y="1em",
                bg="rgba(255,255,255, 0.97)",
            ),
            position="sticky",
            width="100%",
            top="0px",
            z_index="500",
        ),
        component,
    )
