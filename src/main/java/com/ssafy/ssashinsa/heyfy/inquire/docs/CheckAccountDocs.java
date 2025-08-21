package com.ssafy.ssashinsa.heyfy.inquire.docs;

import com.ssafy.ssashinsa.heyfy.inquire.dto.AccountCheckDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "예금주 계좌 존재 여부 확인", description = "사용자에게 연결된 계좌가 있는지 확인합니다. 반환 값은 boolean입니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "계좌 존재 여부 확인 성공",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountCheckDto.class)
                )
        )
})
public @interface CheckAccountDocs {
}